"""
Pydantic models for validating schema input from JSON/dict.

This module provides validation models for creating GLiNER2 schemas
from JSON or dictionary inputs.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class FieldInput(BaseModel):
    """Validates a single structure field.

    Args:
        name: Field name
        dtype: Data type - 'str' for single value, 'list' for multiple values
        choices: Optional list of valid choices for classification-style fields
        description: Optional description of the field
    """
    name: str = Field(..., min_length=1, description="Field name")
    dtype: Literal["str", "list"] = Field(default="list", description="Data type")
    choices: Optional[List[str]] = Field(default=None, description="Valid choices")
    description: Optional[str] = Field(default=None, description="Field description")

    @field_validator('choices')
    @classmethod
    def validate_choices(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Ensure choices list is not empty if provided."""
        if v is not None and len(v) == 0:
            raise ValueError("choices must contain at least one option")
        return v


class StructureInput(BaseModel):
    """Validates a structure block.

    Args:
        fields: List of field definitions
    """
    fields: List[FieldInput] = Field(..., min_length=1, description="List of fields")


class ClassificationInput(BaseModel):
    """Validates a classification task.

    Args:
        task: Task name
        labels: List of classification labels
        multi_label: Whether multiple labels can be selected
    """
    task: str = Field(..., min_length=1, description="Task name")
    labels: List[str] = Field(..., min_length=2, description="Classification labels")
    multi_label: bool = Field(default=False, description="Multi-label classification")

    @field_validator('labels')
    @classmethod
    def validate_labels(cls, v: List[str]) -> List[str]:
        """Ensure labels are unique and non-empty."""
        if len(v) != len(set(v)):
            raise ValueError("labels must be unique")
        if any(not label.strip() for label in v):
            raise ValueError("labels cannot be empty strings")
        return v


class SchemaInput(BaseModel):
    """Root schema validation model.

    Args:
        entities: List of entity types or dict mapping types to descriptions
        structures: Dict mapping structure names to structure definitions
        classifications: List of classification task definitions
        relations: List of relation types or dict mapping types to config
        events: Dict mapping event types to a role list (e.g. ``["Attacker",
            "Victim", "Place"]``) or to a richer config dict
            ``{"roles": [...], "description": "...", "role_descriptions": {...}}``.
    """
    entities: Optional[Union[List[str], Dict[str, str]]] = Field(
        default=None,
        description="Entity types"
    )
    structures: Optional[Dict[str, StructureInput]] = Field(
        default=None,
        description="Structure definitions"
    )
    classifications: Optional[List[ClassificationInput]] = Field(
        default=None,
        description="Classification tasks"
    )
    relations: Optional[Union[List[str], Dict[str, Dict[str, Any]]]] = Field(
        default=None,
        description="Relation types"
    )
    events: Optional[Dict[str, Union[List[str], Dict[str, Any]]]] = Field(
        default=None,
        description="Event types and their roles"
    )

    @field_validator('entities')
    @classmethod
    def validate_entities(
            cls,
            v: Optional[Union[List[str], Dict[str, str]]]
    ) -> Optional[Union[List[str], Dict[str, str]]]:
        """Validate entities format."""
        if v is None:
            return v

        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("entities list cannot be empty")
            if any(not entity.strip() for entity in v):
                raise ValueError("entity names cannot be empty strings")
            if len(v) != len(set(v)):
                raise ValueError("entity names must be unique")
        elif isinstance(v, dict):
            if len(v) == 0:
                raise ValueError("entities dict cannot be empty")
            if any(not key.strip() for key in v.keys()):
                raise ValueError("entity names cannot be empty strings")

        return v

    @field_validator('structures')
    @classmethod
    def validate_structures(
            cls,
            v: Optional[Dict[str, StructureInput]]
    ) -> Optional[Dict[str, StructureInput]]:
        """Validate structures format."""
        if v is None:
            return v

        if len(v) == 0:
            raise ValueError("structures dict cannot be empty")
        if any(not key.strip() for key in v.keys()):
            raise ValueError("structure names cannot be empty strings")

        return v

    @field_validator('classifications')
    @classmethod
    def validate_classifications(
            cls,
            v: Optional[List[ClassificationInput]]
    ) -> Optional[List[ClassificationInput]]:
        """Validate classifications format."""
        if v is None:
            return v

        if len(v) == 0:
            raise ValueError("classifications list cannot be empty")

        # Check for duplicate task names
        task_names = [cls_task.task for cls_task in v]
        if len(task_names) != len(set(task_names)):
            raise ValueError("classification task names must be unique")

        return v

    @field_validator('relations')
    @classmethod
    def validate_relations(
            cls,
            v: Optional[Union[List[str], Dict[str, Dict[str, Any]]]]
    ) -> Optional[Union[List[str], Dict[str, Dict[str, Any]]]]:
        """Validate relations format."""
        if v is None:
            return v

        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("relations list cannot be empty")
            if any(not rel.strip() for rel in v):
                raise ValueError("relation names cannot be empty strings")
            if len(v) != len(set(v)):
                raise ValueError("relation names must be unique")
        elif isinstance(v, dict):
            if len(v) == 0:
                raise ValueError("relations dict cannot be empty")
            if any(not key.strip() for key in v.keys()):
                raise ValueError("relation names cannot be empty strings")

        return v

    @field_validator('events')
    @classmethod
    def validate_events(
            cls,
            v: Optional[Dict[str, Union[List[str], Dict[str, Any]]]]
    ) -> Optional[Dict[str, Union[List[str], Dict[str, Any]]]]:
        """Validate events format."""
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("events dict cannot be empty")
        for name, config in v.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("event type names cannot be empty strings")
            if isinstance(config, list):
                roles = config
            elif isinstance(config, dict):
                roles = config.get("roles")
                if roles is None:
                    raise ValueError(
                        f"event '{name}' config dict must include a 'roles' list"
                    )
            else:
                raise ValueError(
                    f"event '{name}' must map to a role list or config dict"
                )
            if not isinstance(roles, list) or len(roles) == 0:
                raise ValueError(f"event '{name}' must have at least one role")
            if any(not isinstance(r, str) or not r.strip() for r in roles):
                raise ValueError(f"event '{name}' role names cannot be empty strings")
            if len(set(roles)) != len(roles):
                raise ValueError(f"event '{name}' has duplicate roles")
        return v

    @model_validator(mode='after')
    def validate_at_least_one_section(self) -> 'SchemaInput':
        """Ensure at least one section is provided."""
        if all(
                getattr(self, field) is None
                for field in ['entities', 'structures', 'classifications', 'relations', 'events']
        ):
            raise ValueError(
                "At least one of entities, structures, classifications, "
                "relations, or events must be provided"
            )
        return self
