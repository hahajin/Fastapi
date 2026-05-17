"""
backend/agent/models/schemas.py
Pydantic v2 schemas for the structural engineering AI workspace.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NodeType(str, Enum):
    """Supported structural element types in the scene graph."""
    BEAM = "BEAM"
    COLUMN = "COLUMN"
    SLAB = "SLAB"
    WALL = "WALL"
    SUPPORT = "SUPPORT"
    LOAD = "LOAD"
    JOINT = "JOINT"


class Node(BaseModel):
    """A node in the structural scene graph (element, load, or support)."""
    model_config = ConfigDict(extra="forbid")
    
    id: str = Field(..., description="Unique identifier for the node")
    type: NodeType = Field(..., description="Type of structural element")
    position: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Position in metres, origin at (0,0,0)"
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Element-specific properties: dimensions, material, etc."
    )


class Edge(BaseModel):
    """A connection/relationship between two nodes in the scene graph."""
    model_config = ConfigDict(extra="forbid")
    
    id: str = Field(..., description="Unique identifier for the edge")
    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    type: str = Field(..., description="Type of connection, e.g. 'RIGID_CONNECTION'")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Connection-specific properties"
    )


class SceneGraph(BaseModel):
    """Complete structural model representation for frontend rendering."""
    model_config = ConfigDict(extra="forbid")
    
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis results, validation, reports, extracted entities"
    )
    version: str = Field(default="1.0", description="Scene graph schema version")


class IntentType(str, Enum):
    """User intent categories for the NLP parser."""
    CREATE = "CREATE"
    MODIFY = "MODIFY"
    DELETE = "DELETE"
    QUERY = "QUERY"
    ANALYZE = "ANALYZE"
    UNKNOWN = "UNKNOWN"


class ParsedIntent(BaseModel):
    """Result of intent classification on user input."""
    model_config = ConfigDict(extra="forbid")
    
    intent: IntentType = Field(..., description="Classified user intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    raw_text: str = Field(..., description="Original user message text")
    clarification_needed: bool = Field(
        default=False,
        description="Whether the intent is ambiguous and needs clarification"
    )
    clarification_question: str | None = Field(
        default=None,
        description="Question to ask user if clarification is needed"
    )


class ExtractedEntity(BaseModel):
    """A measurable parameter extracted from user input."""
    model_config = ConfigDict(extra="forbid")
    
    entity_type: str = Field(..., description="Type of entity: dimension, material, load, etc.")
    value: Any = Field(..., description="Extracted value (number, string, etc.)")
    unit: str | None = Field(default=None, description="Unit of measurement if applicable")
    source_span: str = Field(..., description="Original text span this entity came from")


class ValidationResult(BaseModel):
    """Result of model validation against engineering rules."""
    model_config = ConfigDict(extra="forbid")
    
    passed: bool = Field(..., description="Whether the model passed all critical checks")
    errors: list[str] = Field(default_factory=list, description="Critical validation errors")
    warnings: list[str] = Field(default_factory=list, description="Non-critical warnings")


class AnalysisResult(BaseModel):
    """Result of an engineering analysis computation."""
    model_config = ConfigDict(extra="forbid")
    
    analysis_type: str = Field(..., description="Type of analysis performed")
    values: dict[str, float] = Field(
        default_factory=dict,
        description="Numerical results of the analysis"
    )
    units: dict[str, str] = Field(
        default_factory=dict,
        description="Units for each value in the results"
    )
    passed: bool = Field(..., description="Whether analysis results meet criteria")
    details: str = Field(..., description="Human-readable explanation of results")


class AgentState(BaseModel):
    """Persistent state for an agent conversation session."""
    model_config = ConfigDict(extra="forbid")
    
    session_id: str = Field(..., description="Unique session identifier")
    conversation: list[dict] = Field(
        default_factory=list,
        description="List of {role, content} message dicts"
    )
    scene_graph: SceneGraph | None = Field(
        default=None,
        description="Current structural model being built/modified"
    )
    last_intent: ParsedIntent | None = Field(
        default=None,
        description="Most recently parsed user intent"
    )
    step: str = Field(default="init", description="Current pipeline step")
    iteration: int = Field(default=0, description="Number of skill iterations executed")
    last_accessed: float = Field(
        default_factory=lambda: __import__("time").time(),
        description="Unix timestamp of last access for cleanup"
    )


class AgentResponse(BaseModel):
    """Response returned to the frontend after agent processing."""
    model_config = ConfigDict(extra="forbid")
    
    session_id: str = Field(..., description="Session identifier for continued conversation")
    scene_graph: SceneGraph | None = Field(
        default=None,
        description="Updated scene graph to render (if any changes)"
    )
    message: str = Field(..., description="Human-readable response message")
    analysis: AnalysisResult | None = Field(
        default=None,
        description="Analysis results if an analysis was performed"
    )
    validation: ValidationResult | None = Field(
        default=None,
        description="Validation results if validation was performed"
    )
    done: bool = Field(..., description="Whether the agent has completed processing")


# ============================================================================
# Round-trip test for SceneGraph schema validation
# ============================================================================
if __name__ == "__main__":
    import json
    
    # Create a test scene graph
    test_graph = SceneGraph(
        nodes=[
            Node(
                id="column_1",
                type=NodeType.COLUMN,
                position=(0.0, 0.0, 0.0),
                properties={"length_m": 3.0, "width_mm": 300, "height_mm": 300, "material": "concrete"}
            ),
            Node(
                id="beam_1",
                type=NodeType.BEAM,
                position=(0.0, 3.0, 0.0),
                properties={"length_m": 5.0, "width_mm": 200, "height_mm": 400, "material": "steel"}
            ),
        ],
        edges=[
            Edge(
                id="conn_1",
                source_id="column_1",
                target_id="beam_1",
                type="RIGID_CONNECTION",
                properties={}
            )
        ],
        metadata={"source": "test"},
        version="1.0"
    )
    
    # Serialize to JSON
    json_str = test_graph.model_dump_json(indent=2)
    print("Serialized SceneGraph:")
    print(json_str)
    
    # Parse back from JSON
    parsed = SceneGraph.model_validate_json(json_str)
    
    # Verify round-trip integrity
    assert parsed.nodes[0].id == "column_1"
    assert parsed.nodes[1].properties["material"] == "steel"
    assert parsed.edges[0].type == "RIGID_CONNECTION"
    assert parsed.version == "1.0"
    
    print("\n✓ Round-trip validation passed!")
    
    # Test extra="forbid" behavior
    try:
        bad_json = test_graph.model_dump()
        bad_json["unknown_field"] = "should fail"
        SceneGraph.model_validate(bad_json)
        print("✗ extra='forbid' did not work!")
    except Exception as e:
        print(f"✓ extra='forbid' correctly rejected unknown field: {type(e).__name__}")