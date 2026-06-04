from sqlalchemy import inspect, text


def ensure_runtime_schema(engine) -> None:
    """Apply small additive schema fixes for installs without Alembic."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "omr_test" not in tables:
        return

    columns = {col["name"] for col in inspector.get_columns("omr_test")}
    dialect = engine.dialect.name
    if dialect == "postgresql":
        string_type = "VARCHAR"
        json_type = "JSON"
    else:
        string_type = "VARCHAR"
        json_type = "JSON"

    additions = {
        "template_image": string_type,
        "info_fields": json_type,
        "options": "INTEGER",
        "rows_per_block": "INTEGER",
        "student_id_digits": "INTEGER",
    }

    with engine.begin() as conn:
        for name, ddl_type in additions.items():
            if name not in columns:
                conn.execute(text(f"ALTER TABLE omr_test ADD COLUMN {name} {ddl_type}"))
