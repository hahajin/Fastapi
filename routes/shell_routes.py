# routes/shell_routes.py  — FastAPI version of the Flask shell_bp
import json
from typing import List, Tuple

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.database import Node, Shell, SurfaceLoad, get_db
from server.user import User

router = APIRouter(tags=["shells"])


# ─── Helpers (unchanged logic from Flask version) ─────────────────────────────

def check_duplicate_node(db: Session, project_id: int,
                          x: float, y: float, z: float,
                          tolerance: float = 1e-6):
    nodes = db.query(Node).filter_by(project_id=project_id).all()
    for node in nodes:
        if (abs(node.x - x) < tolerance and
                abs(node.y - y) < tolerance and
                abs(node.z - z) < tolerance):
            return node
    return None


def subdivide_shell(db: Session, project_id: int, nodes: list,
                    material_id: int, thickness: float,
                    m: int, n: int) -> Tuple[list, list]:
    if len(nodes) != 4:
        return None

    n0, n1, n2, n3 = nodes
    node_grid = [[None] * (m + 1) for _ in range(n + 1)]
    new_nodes: list = []
    new_shells: list = []

    for i in range(n + 1):
        eta = i / n
        for j in range(m + 1):
            xi = j / m
            p = (
                (1 - xi) * (1 - eta) * np.array([n0.x, n0.y, n0.z]) +
                xi * (1 - eta) * np.array([n1.x, n1.y, n1.z]) +
                xi * eta * np.array([n2.x, n2.y, n2.z]) +
                (1 - xi) * eta * np.array([n3.x, n3.y, n3.z])
            )
            x, y, z = p.tolist()
            existing = check_duplicate_node(db, project_id, x, y, z)
            if existing:
                node_grid[i][j] = existing
            else:
                max_num = db.query(func.max(Node.node_number)).filter_by(
                    project_id=project_id).scalar() or 0
                new_node = Node(project_id=project_id, node_number=max_num + 1,
                                x=x, y=y, z=z)
                db.add(new_node)
                db.flush()   # get id without full commit
                node_grid[i][j] = new_node
                new_nodes.append(new_node)

    max_shell_num = db.query(func.max(Shell.shell_number)).filter_by(
        project_id=project_id).scalar() or 0

    for i in range(n):
        for j in range(m):
            nd0 = node_grid[i][j]
            nd1 = node_grid[i][j + 1]
            nd2 = node_grid[i + 1][j + 1]
            nd3 = node_grid[i + 1][j]
            new_shell = Shell(
                project_id=project_id,
                shell_number=max_shell_num + len(new_shells) + 1,
                node_ids=[nd0.id, nd1.id, nd2.id, nd3.id],
                thickness=thickness,
                material_id=material_id,
            )
            db.add(new_shell)
            new_shells.append(new_shell)

    db.commit()
    return new_shells, new_nodes


# ─── Schema ───────────────────────────────────────────────────────────────────

class ShellSubdivideRequest(BaseModel):
    elementSize: List[int] = [2, 2]


# ─── Route ────────────────────────────────────────────────────────────────────

@router.post("/{shell_id}")
def process_shell(
    shell_id: int,
    data: ShellSubdivideRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    shell = db.get(Shell, shell_id)
    if not shell:
        raise HTTPException(status_code=404, detail="Shell not found")

    project_id = shell.project_id
    node_ids   = shell.node_ids
    original_nodes = [db.get(Node, nid) for nid in node_ids]
    if None in original_nodes:
        raise HTTPException(status_code=404, detail="One or more nodes not found")

    material_id = shell.material_id
    thickness   = shell.thickness
    m, n        = data.elementSize

    result = subdivide_shell(db, project_id, original_nodes, material_id, thickness, m, n)
    if not result:
        raise HTTPException(status_code=400, detail="Only quad shells are supported")

    new_shells, new_nodes = result

    # Delete surface loads on the old shell, then delete shell
    for load in db.query(SurfaceLoad).filter_by(shell_id=shell.id).all():
        db.delete(load)
    db.delete(shell)
    db.commit()

    nodes_data = [
        {"id": nd.id, "x": nd.x, "y": nd.y, "z": nd.z,
         "node_number": nd.node_number}
        for nd in new_nodes
    ]
    quads_data = [
        {"id": s.id, "shell_number": s.shell_number, "node_ids": s.node_ids,
         "thickness": s.thickness, "material_id": s.material_id,
         "mesh_type": "MindlinShell"}
        for s in new_shells
    ]

    return {
        "status": "success",
        "nodes": nodes_data,
        "elements": {"quads": quads_data, "triangles": []},
    }
