"""
backend/agent/skills/validator.py
Skill to validate the SceneGraph against engineering rules.
"""

from agent.skills.base_skill import BaseSkill
from agent.models.schemas import (
    AgentState, SceneGraph, ValidationResult, NodeType
)
from agent.tools.registry import registry


class ValidatorSkill(BaseSkill):
    """Validate SceneGraph against structural rules and constraints."""
    
    name = "validator"
    description = "Validate model integrity and engineering constraints"
    
    async def run(self, state: AgentState) -> AgentState:
        if state.scene_graph is None:
            state.scene_graph = SceneGraph()
        
        errors = []
        warnings = []
        graph = state.scene_graph
        
        # Build node lookup
        node_ids = {node.id for node in graph.nodes}
        node_positions = {node.id: node.position for node in graph.nodes}
        
        # Rule 1: Edges reference existing nodes
        for edge in graph.edges:
            if edge.source_id not in node_ids:
                errors.append(f"Edge '{edge.id}' references unknown source node: {edge.source_id}")
            if edge.target_id not in node_ids:
                errors.append(f"Edge '{edge.id}' references unknown target node: {edge.target_id}")
        
        # Rule 2: No duplicate positions
        positions_seen = {}
        for node in graph.nodes:
            pos_key = tuple(round(c, 3) for c in node.position)
            if pos_key in positions_seen:
                warnings.append(
                    f"Nodes '{positions_seen[pos_key]}' and '{node.id}' share similar positions: {pos_key}"
                )
            else:
                positions_seen[pos_key] = node.id
        
        # Rule 3: BEAM/COLUMN required properties
        required_props = {"length_m", "width_mm", "height_mm", "material"}
        for node in graph.nodes:
            if node.type in [NodeType.BEAM, NodeType.COLUMN]:
                missing = required_props - set(node.properties.keys())
                if missing:
                    errors.append(
                        f"{node.type} '{node.id}' missing required properties: {missing}"
                    )
        
        # Tool call 1: Check span-to-depth ratio for beams
        for node in graph.nodes:
            if node.type == NodeType.BEAM:
                props = node.properties
                if "length_m" in props and "height_mm" in props:
                    depth_m = props["height_mm"] / 1000
                    result = await registry.execute(
                        "check_span_to_depth_ratio",
                        {"span_m": props["length_m"], "depth_m": depth_m, "element_type": "BEAM"}
                    )
                    if not result.get("passed"):
                        warnings.append(
                            f"Beam '{node.id}': span/depth ratio {result.get('ratio')} exceeds limit {result.get('limit')}"
                        )
        
        # Tool call 2: Estimate self-weight for all structural elements
        for node in graph.nodes:
            if node.type in [NodeType.BEAM, NodeType.COLUMN, NodeType.SLAB, NodeType.WALL]:
                props = node.properties
                if "length_m" in props and "width_mm" in props and "height_mm" in props:
                    # Approximate volume
                    vol = props["length_m"] * (props["width_mm"]/1000) * (props["height_mm"]/1000)
                    material = props.get("material", "concrete")
                    weight_result = await registry.execute(
                        "estimate_self_weight",
                        {"volume_m3": vol, "material": material}
                    )
                    node.properties["self_weight_kN"] = weight_result["weight_kN"]
        
        # Assemble validation result
        validation = ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
        # Store in metadata
        graph.metadata["validation"] = validation.model_dump()
        
        # Update step if critical failures
        if not validation.passed and errors:
            state.step = "validation_failed"
        
        return state