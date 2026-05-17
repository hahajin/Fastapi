"""
backend/tools/registry.py
Tool registry with decorator for LLM function calling.
"""

import inspect
import logging
from functools import wraps
from typing import Any, Callable

from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)


def _type_to_json_schema_type(py_type: type) -> dict:
    """Convert Python type hints to JSON Schema type definitions."""
    if py_type == str:
        return {"type": "string"}
    elif py_type == int:
        return {"type": "integer"}
    elif py_type == float:
        return {"type": "number"}
    elif py_type == bool:
        return {"type": "boolean"}
    elif py_type == dict:
        return {"type": "object"}
    elif py_type == list:
        return {"type": "array"}
    elif hasattr(py_type, "__origin__"):
        # Handle Optional, Union, etc.
        origin = py_type.__origin__
        if origin is list:
            args = py_type.__args__
            return {"type": "array", "items": _type_to_json_schema_type(args[0]) if args else {}}
        elif origin is dict:
            return {"type": "object"}
    return {"type": "string"}  # Fallback


def tool(name: str, description: str):
    """
    Decorator to register an async function as an LLM-callable tool.
    
    Automatically generates OpenAI-format tool schema from function signature.
    """
    def decorator(fn: Callable):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)
        
        # Build tool schema from signature
        sig = inspect.signature(fn)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            param_desc = f"Parameter: {param_name}"  # Could extract from docstring
            
            properties[param_name] = {
                **_type_to_json_schema_type(param_type),
                "description": param_desc
            }
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        wrapper.tool_schema = {  # type: ignore
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
            }
        }
        wrapper.tool_name = name  # type: ignore
        return wrapper
    return decorator


class ToolRegistry:
    """Registry for LLM-callable tools with schema generation and execution."""
    
    def __init__(self):
        self._tools: dict[str, Callable] = {}
    
    def register(self, fn: Callable) -> Callable:
        """Register a decorated tool function."""
        if not hasattr(fn, "tool_schema"):
            raise ValueError(f"Function {fn.__name__} must be decorated with @tool")
        self._tools[fn.tool_name] = fn  # type: ignore
        logger.info(f"Registered tool: {fn.tool_name}")  # type: ignore
        return fn
    
    def get_schemas(self) -> list[dict]:
        """Return all tool schemas for LLM function calling."""
        return [fn.tool_schema for fn in self._tools.values()]  # type: ignore
    
    async def execute(self, name: str, arguments: dict) -> Any:
        """Execute a registered tool by name with given arguments."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        fn = self._tools[name]
        logger.debug(f"Executing tool '{name}' with args: {arguments}")
        result = await fn(**arguments)
        logger.debug(f"Tool '{name}' returned: {result}")
        return result


# ============================================================================
# Built-in Engineering Tools
# ============================================================================

@tool(
    name="search_engineering_standards",
    description="Search for relevant engineering code provisions or standards by query."
)
async def search_engineering_standards(query: str) -> str:
    """Stub: Search engineering standards database."""
    return f"No results found for: {query}"


@tool(
    name="compute_section_properties",
    description="Compute geometric properties of a rectangular cross-section."
)
async def compute_section_properties(
    width_mm: float,
    height_mm: float,
    shape: str = "rectangular"
) -> dict:
    """Compute area, moments of inertia, and section moduli for rectangular sections."""
    if shape != "rectangular":
        return {"error": f"Shape '{shape}' not supported"}
    
    # Convert mm to m for calculations
    b = width_mm / 1000
    h = height_mm / 1000
    
    area = b * h  # m²
    ix = (b * h**3) / 12  # m⁴ (about x-axis, horizontal)
    iy = (h * b**3) / 12  # m⁴ (about y-axis, vertical)
    sx = ix / (h / 2) if h > 0 else 0  # m³
    sy = iy / (b / 2) if b > 0 else 0  # m³
    
    return {
        "area_m2": round(area, 6),
        "ix_m4": round(ix, 8),
        "iy_m4": round(iy, 8),
        "sx_m3": round(sx, 6),
        "sy_m3": round(sy, 6)
    }


@tool(
    name="check_span_to_depth_ratio",
    description="Check if beam span-to-depth ratio meets simplified code limits."
)
async def check_span_to_depth_ratio(
    span_m: float,
    depth_m: float,
    element_type: str
) -> dict:
    """Check span/depth ratio against simplified limits by element type."""
    if depth_m <= 0:
        return {"ratio": float('inf'), "limit": 0, "passed": False, "error": "Invalid depth"}
    
    ratio = span_m / depth_m
    
    # Simplified limits (would be code-specific in production)
    limits = {
        "BEAM": 20,
        "COLUMN": 15,
        "SLAB": 30,
        "WALL": 25
    }
    limit = limits.get(element_type, 20)
    
    return {
        "ratio": round(ratio, 2),
        "limit": limit,
        "passed": ratio <= limit
    }


@tool(
    name="estimate_self_weight",
    description="Estimate self-weight of an element given volume and material."
)
async def estimate_self_weight(volume_m3: float, material: str) -> dict:
    """Estimate self-weight using material density lookup."""
    # Densities in kN/m³
    densities = {
        "steel": 78.5,
        "concrete": 25.0,
        "timber": 6.0,
        "aluminum": 27.0
    }
    
    density = densities.get(material.lower(), 25.0)  # Default to concrete
    weight_kn = volume_m3 * density
    
    return {"weight_kN": round(weight_kn, 3)}


@tool(
    name="get_load_combination",
    description="Get load combination factors for a given code and combination type."
)
async def get_load_combination(code: str, combo_type: str) -> dict:
    """Stub: Return load combination factors."""
    # Simplified Eurocode ULS combination
    if code.lower() == "eurocode" and combo_type.lower() == "uls":
        return {"factors": {"dead": 1.35, "live": 1.5}}
    
    # Default fallback
    return {"factors": {"dead": 1.0, "live": 1.0}}


# ============================================================================
# Module-level registry instance with built-in tools
# ============================================================================

registry = ToolRegistry()
registry.register(search_engineering_standards)
registry.register(compute_section_properties)
registry.register(check_span_to_depth_ratio)
registry.register(estimate_self_weight)
registry.register(get_load_combination)