# ADR-0005 — Dashboard JSONB schema for axes

**Status:** Accepted
**Date:** 2026-05-10

## Context

Legacy `kaitakuai/dashboard` stores image metadata with typed Postgres columns (`gpu`, `model` enums). Adding a new axis (e.g., `quant`) requires alembic migration + pydantic schema update + UI update — friction every time we extend the architecture.

Per ADR-0003, the system is designed to grow new axes cheaply (1 line in `tools/naming.cue`). Dashboard schema should match this property.

## Decision

Dashboard image table uses **JSONB column `axes`** + `axes_catalog` registry table. The catalog is populated by sync script from `tools/naming.cue::axes`.

```sql
ALTER TABLE images
  ADD COLUMN axes JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN versions JSONB NOT NULL,
  ADD COLUMN base_kind TEXT NOT NULL,
  ADD COLUMN gpu TEXT GENERATED ALWAYS AS (axes->>'gpu') STORED,
  ADD COLUMN model TEXT GENERATED ALWAYS AS (axes->>'model') STORED;

CREATE INDEX images_gpu_idx ON images(gpu);
CREATE INDEX images_model_idx ON images(model);
CREATE INDEX images_axes_idx ON images USING GIN(axes);

CREATE TABLE axes_catalog (
  axis_name TEXT PRIMARY KEY,
  axis_type TEXT NOT NULL,             -- 'identity' or 'runtime'
  prefix TEXT,
  description TEXT,
  allowed_values JSONB,
  display_order INT
);
```

Stable axes (`gpu`, `model`) are extracted as generated columns for index-friendly queries; everything else lives in JSONB.

UI filter chips driven by `axes_catalog` — adding a new axis updates the table on next sync, UI shows new filter automatically.

## Consequences

- **Adding a new axis = 0 alembic migrations** in dashboard
- **JSONB GIN index** keeps filter performance acceptable up to ~10K images
- **Generated columns** for hot axes (gpu, model) preserve type-safety at the SQL boundary

## Alternatives considered

- **Pure typed columns** (legacy): cited above — schema migration on every axis
- **Separate axes table** (one row per (image, axis_value)): more normalized but heavier joins for typical "list all images" queries
