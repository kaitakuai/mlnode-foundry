# Архитектура `kaitakuai/mlnode-foundry`

> Архитектура build-системы образов ML-ноды.

## 1. Цели и принципы

| # | Принцип | Зачем |
|---|---------|-------|
| P1 | **4 явных слоя в pipeline** с независимыми артефактами в GHCR | Кеширование и переиспользование между сборками; не пересобираем ML-зависимости 8 раз для 8 GPU |
| P2 | **Профиль — единственный variant-input** | Имя пакета, тэг, registry-метаданные выводятся из него; убирает класс багов «опечатка в тэге» |
| P3 | **Maintenance > implementation** | На команде из 2 dev-ов с дефицитом времени критично минимизировать ручной труд при bump vLLM / новой модели / новой GPU |
| P4 | **Build не блокируется на бенчмарках** | Дорогая GPU-валидация — асинхронный side-channel, не часть CI-критпуть |
| P5 | **Расширяемость по осям** | identity / runtime разделены явно; новая ось ≈ строка в `tools/naming.cue` |
| P6 | **Supply chain by default** | SLSA L3 + SBOM + cosign keyless signing на каждый publish, без opt-in |
| P7 | **Финальные Dockerfile-ы не коммитим** | Render по требованию + attestation на каждый publish; отсутствие drift |
| P8 | **Custom thin DSL** поверх `docker buildx bake` | На масштабе ≤150 профилей и команды 2 dev — оптимально по cost/value |
| P9 | **Greenfield repo** для рефакторинга, со заморозкой старого | Минимизирует риск для production (нод-операторы пинят digest-ы из старого репо); параллельная разработка без конфликтов; тривиальный откат |
| P10 | **Spec / state / policy разделены** на всех уровнях | Профиль — pure spec (intent человека); observed state (validation results, benchmarks, vLLM ENV snapshot) — отдельно в `state/`; naming + tag + axes policy — глобально в `tools/naming.cue`; pin upstream — immutable lock в `tools/stage3.lock.cue`. Не смешиваем «что хотим», «что произошло» и «как именуем» в одном файле — **ни на per-profile, ни на org-wide уровне** |
| P11 | **[Cue](https://cuelang.org/) для типизированных конфигов и схем** | Единый язык для **human-authored intent**: профили, naming policy, runner inventory, schemas. Sum types (`#BaseProfile \| #OverlayProfile`) — нативно. Unification (`&`) вместо deep-merge — коммутативна, конфликты падают громко. Constraints (`>=1`, `=~ regex`) проверяются compiler-ом. TOML — только для внешних форматов (`pyproject.toml`); YAML — только где tooling требует (GitHub Actions). JSON — для **machine-written observed state** (валидируется Cue-схемами через `cue vet`) |
| P11a | **Boundary Cue ↔ JSON ↔ Python** | **Cue** — то что пишет человек (intent, schemas, policy). **JSON** — то что пишет машина (observed state, BuildKit attestations, OCI labels), валидируется Cue-схемами. **Python** — orchestration: subprocess-вызовы (cue, docker, cosign), file I/O, content hashing, template generation. Каждый язык на своём месте; не заменяют друг друга |
| P12 | **Composition через unification, не inheritance** | `extends:` уходит — заменяется Cue-импортами и `&`. Order-independent (unification коммутативна), type-safe (compiler ловит несовместимые значения), безопасно глубоко по необходимости. Style-конвенция: max 2 базы на профиль, base-файлы атомарны — но это **stylistic**, не safety-invariant |
| P13 | **CLI как нативный Python-пакет** (Typer) | Вместо Makefile — distributable `mlnode-foundry` CLI: type-checked аргументы, auto-generated help, autocomplete, subcommand tree. Устанавливается через `pip install -e .` или `pip install mlnode-foundry`; внешние пользователи могут запускать команды без клонирования репо |

## 2. Цепочка слоёв

```
Stage 0   vllm/vllm-openai:<vllm-ver>                                              [внешний]
            │ residual sampler-stack fork (репо kaitakuai/vllm)
            ▼
Stage 1   ghcr.io/kaitakuai/vllm-sampler-residual:<vllm-ver>-k<rev>              [репо kaitakuai/vllm]
            │ pip install gonka-poc + upstream packages/api/Dockerfile (target=app)
            ▼
Stage 2   ghcr.io/kaitakuai/vllm-poc:<vllm-ver>-k<rev>                            [репо kaitakuai/vllm]
            │ apt + OpenSSL + poetry + venv + mlnode source
            ▼
Stage 3   ghcr.io/kaitakuai/mlnode-base:<mlnode-ver>-vllm<vllm-ver>-k<rev>        [этот репо, новый артефакт]
            │ hw-patches + runner-patch + ENV — из профиля
            ▼
Stage 4   ghcr.io/kaitakuai/mlnode-<gpu>-<model>:<tag>                           [этот репо, потребительский образ]
```

| Слой | Где собирается | Что внутри | Когда ребилдится |
|------|----------------|------------|------------------|
| **Stage 0** | вне нашей зоны (vLLM Project) | upstream vLLM-OpenAI Docker image | релизы vLLM |
| **Stage 1** | репо `kaitakuai/vllm` | upstream + thin sampler-residual fork (ADR-0014) | bump vLLM или residual-патчей |
| **Stage 2** | репо `kaitakuai/vllm` | Stage 1 + `pip install gonka-poc` (ADR-0013) | bump Stage 1 digest или gonka-poc |
| **Stage 3** | репо `kaitakuai/mlnode-foundry` | apt + OpenSSL + poetry + venv + mlnode source — **GPU-агностик** | bump Stage 2 digest, upstream commit или `patches/` |
| **Stage 4** | репо `kaitakuai/mlnode-foundry` | Stage 3 + hw-patches + runner-patch + ENV из профиля | правка профиля, hw-патчей, runner-патчей или Stage 3 digest |

**Линий `full` и `overlay` нет.** Overlay-режим (поверх upstream `product-science/mlnode` бинарника) сохранён как `naming.mode: upstream-overlay` в профиле — пропускает Stage 3, идёт через тот же Stage 4 с `BASE_IMAGE=ghcr.io/product-science/mlnode:<ver>`. Тот же Dockerfile, тот же набор патчей.

## 3. Два типа осей

Любое значение в профиле относится к одному из двух типов; граница объявлена в [ADR-0003](./adr/0003-profile-dsl-and-axis-types.md) и проверяется JSON-Schema.

| Тип | Что меняет | Где живёт | Примеры |
|-----|------------|-----------|---------|
| **identity** | другой образ (новый GHCR-coord или другой digest при том же coord) | `identity.axes` в профиле | `gpu`, `model`, `framework`, `quant`, `build_mode` |
| **runtime** | НЕ меняет образ; vLLM CLI-флаг или ENV при `docker run` | блок `runtime_defaults` в профиле; OCI label `gonka.kaitaku.runtime_defaults` на образе | `tp_layout`, `context_length`, `gpu_memory_utilization`, любые vLLM-флаги |

**Правило теста**: если ось требует другого содержимого в образе (другие пакеты, другие патчи, другой ENV), это **identity**. Если ось — это просто аргумент при запуске контейнера, это **runtime**.

Категория «tuning» (тот же образ, но новый тэг) была отвергнута как искусственная: на практике если оси не меняют содержимое — нет причины публиковать второй образ под другим тэгом. Лучше один образ + override через runtime-флаг при запуске.

## 4. Реестр осей и глобальная политика именования

Один файл `tools/naming.cue` объединяет реестр identity-осей и политику composition имени и тэга. **Чтобы изменить конвенцию (например, добавить `framework` в имя пакета вместо тэга) — правится один файл, не 50 профилей.**

Файл одновременно является **схемой** (через `#`-определения) и **значением** (через concrete `axes:`, `package:`, `tag:` блоки).

```cue
// tools/naming.cue
package naming

#AxisType: "identity" | "tuning" | "runtime"

#Axis: {
    type:           #AxisType
    description:    string
    prefix?:        string                    // обязателен для tag-axes
    name_axis_only?: bool                     // true → только в имени пакета, не в теге
    status?:        "active" | "reserved"
    default?:       string
    allowed_values?: [...string]
}

axes: [string]: #Axis

axes: gpu:        {type: "identity", name_axis_only: true, description: "GPU model (h100, b300, ...)"}
axes: model:      {type: "identity", name_axis_only: true, description: "Model family (qwen, kimi, minimax, ...)"}
axes: quant:      {type: "identity", prefix: "q", description: "Quantization scheme (int4, fp8, nvfp4, ...)"}
axes: framework:  {type: "identity", prefix: "f", description: "Inference engine (vllm, sglang, ...)"}
axes: build_mode: {type: "identity", prefix: "m", description: "inference | inference+train"}
axes: transform: {
    type:        "identity"
    prefix:      "t"
    status:      "reserved"
    default:     "full"
    allowed_values: ["full", "slim"]
    description: "Reserved for Stage 5 image minimization"
}

package: {
    prefix: "ghcr.io/kaitakuai/mlnode"
    axes: ["gpu", "model"]                    // → mlnode-<gpu>-<model>
}

tag: {
    axes_order: ["quant", "framework", "transform"]
    axis_format: "-{prefix}.{value}"
    modes: {
        "kaitakuai-base":   "<mlnode>-vllm<vllm>{tag_axes}-k<rev>"
        "upstream-overlay": "<upstream>-overlay{tag_axes}-k<rev>"
    }
}
```

Профили импортируют этот пакет — Cue compiler проверяет на этапе валидации что профиль использует только зарегистрированные оси.

`tp_layout`, `context_length`, `gpu_memory_utilization` и прочие vLLM-флаги — **не identity-оси**. Они живут в `runtime_defaults` блоке профиля и пишутся как OCI label на образе; нод-оператор может их менять при `docker run` без переcборки образа.

### Тэг-формат

| Mode | Шаблон | Пример |
|------|--------|--------|
| `kaitakuai-base` | `<mlnode>-vllm<vllm>(-<prefix>.<value>)*-k<rev>` | `0.2.13-vllm0.20.0-q.int4-k1` |
| `upstream-overlay` | `<upstream>-overlay(-<prefix>.<value>)*-k<rev>` | `3.0.13-alpha5-overlay-k1` |

**Парсинг в dashboard — по prefix-разделителю**, не по позиции. Добавление новой оси не ломает старые тэги.

### Имя пакета

Имя пакета — `ghcr.io/kaitakuai/mlnode-<gpu>-<model>`. Стандарт жёстко фиксирован в `naming.yaml`:

- `mlnode-b300-kimi`
- `mlnode-h100-qwen`
- `mlnode-b200-minimax`

## 5. Профиль (DSL) — pure spec на Cue

Один `.cue` файл на уникальную identity-комбинацию. Профиль — **pure spec**: декларация intent, который пишет человек. Никаких observed-state полей, никакой per-profile naming policy.

### 5.1 Сами схемы — sum type

Mode-свич реализован как **discriminated union** через native Cue:

```cue
// profiles/_schema.cue
package profiles

import "github.com/kaitakuai/mlnode-foundry/tools/naming"

#Identity: {
    axes: {
        gpu:        =~ "^[a-z][a-z0-9]+$"
        model:      =~ "^[a-z][a-z0-9]+$"
        quant?:     "int4" | "fp8" | "nvfp4" | "awq4bit" | "mxfp4"
        framework?: "vllm" | "sglang" | "trtllm"
        transform?: "full" | "slim"
        ...
    }
    version: { rev: int & >=1, ... }
}

#CommonProfile: {
    identity: #Identity
    hw_patches: [...string]
    runner_patch: string
    env: [string]: string
    runtime_defaults: {...}
    validation_targets: {
        smoke:     [...string]
        benchmark: [...string]
    }
    description: string
    notes?: string
    minimization?: {...}    // reserved для Stage 5
}

#BaseProfile: #CommonProfile & {
    mode: "kaitakuai-base"
    identity: version: {
        mlnode: =~ "^[0-9]+\\.[0-9]+\\.[0-9]+"
        vllm:   =~ "^[0-9]+\\.[0-9]+\\.[0-9]+"
        rev:    int & >=1
    }
    // base.image+digest НЕ допускаются — резолвятся из tools/stage3.lock.cue
}

#OverlayProfile: #CommonProfile & {
    mode: "upstream-overlay"
    identity: version: {
        upstream: string
        rev:      int & >=1
    }
    base: {
        image:            =~ "^ghcr.io/"
        digest:           =~ "^sha256:[a-f0-9]{64}$"
        upstream_version: string
    }
}

#Profile: #BaseProfile | #OverlayProfile
```

Cue compiler проверяет инвариант: если `mode == "kaitakuai-base"`, то `base.image` запрещён; если `mode == "upstream-overlay"`, то `base.image+digest+upstream_version` обязательны. **Никакого ручного `if/then` в JSON-Schema — это нативная типизация.**

### 5.2 Конкретный профиль

```cue
// profiles/b300-kimi-int4.cue
package profiles

import (
    "github.com/kaitakuai/mlnode-foundry/profiles/_base"
)

profile: #BaseProfile & _base.b300 & _base.kimi_int4 & {
    identity: {
        axes: {
            gpu:   "b300"
            model: "kimi"
            quant: "int4"
        }
        version: {
            mlnode: "0.2.13"
            vllm:   "0.20.0"
            rev:    1
        }
    }
    runner_patch: "b300-kimi"
    description: "B300 Blackwell Ultra + Kimi-K2.6 INT4 (4×B300 SXM)"
    notes: """
        Why VLLM_USE_FLASHINFER_MOE_INT4=1: +138% throughput vs Marlin on Blackwell sm_100.
        Evidence: experiments/2026-04/kimi-k26-b300-eager-flashinfer/
        """
}
```

`_base.b300` определяет hw-patches и базовые ENV; `_base.kimi_int4` определяет model-specific ENV (`VLLM_USE_FLASHINFER_MOE_INT4`, `POC_BATCH_SIZE_DEFAULT`) и runtime_defaults для INT4. Профиль **унифицирует** оба — Cue compiler ловит конфликты.

### 5.3 Что Cue даёт нам относительно TOML+JSON-Schema

| Аспект | TOML+JSON-Schema (что было) | Cue (что есть) |
|--------|---------------------------|----------------|
| Sum types (mode-discriminated union) | `oneOf:`/`if:`/`then:` в JSON-Schema; ручная коррекция Python-валидатором | `#Profile: #BaseProfile \| #OverlayProfile` — нативно |
| Composition между profile + bases | Python deep-merge (order matters, конфликты молча перезаписываются) | Unification (`&`) — коммутативна, конфликты падают громко |
| Constraints (rev >=1, regex на digest) | JSON-Schema конструкции (`pattern`, `minimum`) | Inline в типе (`int & >=1`, `=~ "regex"`) |
| Дубликат `mode` ↔ `base.kind` | Ручная invariant-проверка | **Невозможен** — `base` структура зависит от `mode` через type union |
| Schema и data в одном языке | две системы (TOML data + JSON-Schema), две парсинг-цепочки | один язык, одна цепочка |
| Order-independence в composition | нет (deep-merge порядок-зависимый) | Да (unification коммутативна и ассоциативна) |

D2 (extends-footguns) теперь закрыт **более глубоко**: это не просто «один уровень глубины», а **type-safe composition** где конфликты невозможно skрыть.

### 5.4 Style-конвенция (не safety-invariant)

Хотя Cue безопасно справится с глубокой композицией — для **читаемости человеком** держим конвенцию:

- **Max 2 базы на профиль** (`_base.b300 & _base.kimi_int4`) — больше становится трудно отслеживать откуда что приходит
- **Base-файлы атомарны** — не импортируют друг друга
- **Базы определяют только common subset** — специфичные значения остаются в leaf-профиле

Это **читабельность**, а не безопасность. Cue compiler не fail-нет если нарушите, но `mlnode-foundry profile lint` посветит warning.

### Что НЕ хранится в профиле

| Что | Где живёт | Почему не в профиле |
|-----|-----------|---------------------|
| `package_axes` / `tag_axes` | `tools/naming.cue` (глобально) | org-wide policy, не per-profile decision |
| `status`, `validation.<tier>.result` | `state/<package-tag>.json` (см. §6) | observed state, пишется автоматикой, не человеком |
| `metrics.expected_nonces_per_min` | `state/<package-tag>.json` | observed state, обновляется агентом после бенча |
| Runtime override на конкретном хосте | при `docker run` (env vars / args) | per-deployment, не per-image |

### Inheritance (`extends:`)

Профиль может наследоваться от файлов в `profiles/_base/`. Resolution rules:

- Deep merge, later wins.
- Lists сливаются по name (для `hw_patches`) или конкатенируются.
- Базы валидируются по той же схеме, что и финальные профили.
- Базы не публикуются как образы (флаг `_base: true` в файле).

Каталог `profiles/_base/` стартует пустым; наполняется по факту первой общности (например, после третьего `b300-*` профиля имеет смысл вынести `_base/b300.cue`).

### Self-contained snapshot

Команда `tools/expand-profile.py profiles/<name>.cue` резолвит все ссылки (extends, hw_patches, runner_patch, axes catalog, stage3 digest) в инлайн-формат. Полезно для:

- Аудита (один файл вместо обхода репо)
- Архивации release-snapshot-а
- Отправки внешнему коллеге без доступа к репо

## 6. Observed state и lifecycle

State (что произошло с образом — собрался ли, прошли ли smoke / benchmark) **не живёт в профиле**. Профиль — это intent, state — это наблюдение. Они в разных файлах, потенциально в разных репо.

### `state/<package-tag>.json` — observed state (data в JSON, schema в Cue)

Файл создаётся и обновляется автоматически: build-CI пишет результат сборки, benchmark-агент пишет результаты валидации. **Format — JSON**, потому что state это machine-written data; schema живёт в Cue (`state/_schema.cue`); валидация — `cue vet state/<x>.json state/_schema.cue`.

**Почему JSON а не Cue:**

- Cue — язык для **human-authoring** (composition, types, constraints). Машина не нуждается в этих фичах при записи.
- JSON универсален: `jq`, `json.dumps()`, любой агент с любого языка пишет тривиально.
- Cue нативно валидирует JSON через `cue vet` — нет потери type-safety.
- BuildKit и cosign генерят JSON-attestations нативно — выравниваемся с upstream форматом.
- Уменьшаем зависимости агентов: не нужен `cue` binary в каждом workflow, который пишет state.

```json
// state/mlnode-b300-kimi-0.2.13-vllm0.20.0-q.int4-k1.json
{
  "profile": "profiles/b300-kimi-int4.cue",
  "profile_hash": "a1b2c3...",
  "status": "validated",
  "image": {
    "package": "ghcr.io/kaitakuai/mlnode-b300-kimi",
    "tag": "0.2.13-vllm0.20.0-q.int4-k1",
    "digest": "sha256:...",
    "built_at": "2026-05-10T14:32:00Z",
    "cosign_identity": "https://github.com/kaitakuai/mlnode-foundry/.github/workflows/build-stage4.yml@refs/heads/main"
  },
  "validation": {
    "build_smoke": {"result": "pass", "at": "2026-05-10", "commit": "a1b2c3"},
    "gpu_smoke":   {"result": "pass", "at": "2026-05-11", "instance": "vast/B300-1x"},
    "benchmark":   {"result": "pending"}
  },
  "metrics": {
    "expected_nonces_per_min": 5120,
    "measured_at": "2026-05-12",
    "cost_estimate_usd": 12.40,
    "benchmark_report": "experiments/2026-05/b300-kimi-int4-k1/"
  }
}
```

Schema-файл `state/_schema.cue` остаётся **в Cue** — он human-authored и описывает структуру + constraints:

```cue
// state/_schema.cue
package state

#State: {
    profile:      string & =~ "^profiles/.+\\.cue$"
    profile_hash: =~ "^[a-f0-9]{64}$"
    status:       "draft" | "validated" | "benchmarked" | "deprecated"
    image: {
        package:         =~ "^ghcr\\.io/"
        tag:             string
        digest:          =~ "^sha256:[a-f0-9]{64}$"
        built_at:        string  // ISO 8601
        cosign_identity: =~ "^https://github\\.com/"
    }
    validation: [string]: {
        result: "pass" | "fail" | "pending"
        at?:    string
        ...
    }
    metrics: {
        expected_nonces_per_min?: int & >0
        measured_at?:             string
        cost_estimate_usd?:       number & >=0
        benchmark_report?:        string
    }
}
```

CI вызывает `cue vet state/<x>.json state/_schema.cue` чтобы проверить соответствие схеме. **Type-safety та же**, что была бы при native-Cue, но без burden-а на машину писать в Cue-формате.

### Lifecycle

```
   draft  ─── Tier 2 pass ───►  validated  ─── Tier 3 pass ───►  benchmarked
                                    │                                 │
                                    └────────── deprecated ◄──────────┘
```

| State | Что значит | Когда используется |
|-------|------------|---------------------|
| `draft` | собрался, реальное железо не проверялось | researcher-experiments OK; **не для прода** |
| `validated` | Tier 2 (real-GPU smoke) прошёл | минимально пригодно; нод-операторы могут пробовать |
| `benchmarked` | Tier 3 (full benchmark) прошёл, числа в `metrics:` | подтверждённые числа, рекомендуется для прода |
| `deprecated` | заменён более новой версией | dashboard скрывает по умолчанию |

Образ публикуется в любом state-е. Dashboard читает state-файлы и показывает разные бейджи. Нод-операторы фильтруют `status >= validated`.

### Почему spec / state разделены

- **Чистый `git blame` на профилях**: только человеческие изменения, без шума от агента
- **Профиль — immutable от системы**: build-pipeline и agent его не трогают; они пишут только в `state/`
- **Reset state без правки spec**: если что-то пошло не так, можно `rm state/<x>.json` и пересобрать без касания профиля
- **Multi-instance state возможно**: если один и тот же образ валидируется на разных runner-ах, можно завести `state/<tag>/<runner>.json` без размножения профилей

### Где физически живут state-файлы

Два варианта (выбираем в PR #2):

| Вариант | Pro | Contra |
|---------|-----|--------|
| **A.** В том же репо `mlnode-foundry/state/` (committed) | git history валидаций; dashboard читает напрямую | агент пушит в репо — каждый бенч = коммит, шумит history |
| **B.** В отдельной ветке `state` или отдельном репо `mlnode-foundry-state` | spec-репо чистый | две точки правды, навигация сложнее |

**Дефолт — вариант A** с настройкой: коммиты от агента имеют префикс `state:` и исключаются из основного `git log` через alias. Dashboard читает напрямую из `main` ветки.

> Замечание: ошибки в state-файлах **не критичны** — это справочная информация. Race conditions при concurrent-записи допустимы; один потерянный benchmark-результат не сломает систему. Контракт — eventual consistency без транзакций.

## 7. Validation tiers

| Tier | Проверка | Где запускается | Стоимость | Триггер |
|------|----------|-----------------|-----------|---------|
| **0 — Static** | JSON-schema, axes catalog, render Dockerfile, lint, drift detection | GH Actions ubuntu-latest | бесплатно | каждый PR/push |
| **1 — Build-only smoke** | Stage 4 собирается, entrypoint exists, `import vllm` не падает (CPU-only) | GH Actions ubuntu-latest | бесплатно | каждый PR с правкой профиля или его зависимостей |
| **2 — Real-GPU smoke** | образ запускается, `/api/v1/inference/up` отвечает, 1-10 nonces | vast.ai эфемера или ssh-host | $0.50-3 | PR-метка `/validate-gpu` или первая публикация профиля |
| **3 — Full benchmark** | 1000+ nonces, logprobs, cross-validation, итоговые метрики | vast.ai / ssh-host через `poc-benchmark` AI-агента | $5-50 | PR-метка `/benchmark`, cron, или bump major-версии |

**Tier 2 и 3 — opt-in**, не дефолт CI. Build-система не блокируется на дорогих проверках. Это критично — иначе бюджет на validation выгорит за неделю.

## 8. Runner-абстракция

Один файл `tools/runners.cue` содержит весь inventory GPU runner-ов:

```cue
// tools/runners.cue
package runners

#Budget: {
    max_cost_per_run_usd?: float & >0
    max_runtime_min:        int & >0
    monthly_usd?:           float & >0
}

#VastRunner: {
    type:        "vast.ai"
    gpu_query:   string
    disk_gb:     int & >=100
    ssh_key_env: string
    budget:      #Budget
}

#SSHRunner: {
    type:        "ssh"
    host_env:    string
    user:        string
    ssh_key_env: string
    budget:      #Budget
}

#Runner: #VastRunner | #SSHRunner

runners: [string]: #Runner

runners: "vast-b300-4x": {
    type: "vast.ai"
    gpu_query: "RTX_PRO_6000_SE_x4"
    disk_gb: 500
    ssh_key_env: "VAST_SSH_KEY"
    budget: {max_cost_per_run_usd: 10, max_runtime_min: 90, monthly_usd: 200}
}

runners: "cherry-b300-8x": {
    type: "ssh"
    host_env: "CHERRY_B300_HOST"
    user: "ubuntu"
    ssh_key_env: "CHERRY_SSH_KEY"
    budget: {max_runtime_min: 120}
}
```

Профили ссылаются на runner по name (Cue compiler проверяет существование в `runners.cue`):

```cue
profile: ... & {
    validation_targets: {
        smoke:     ["vast-b300-1x"]
        benchmark: ["vast-b300-4x", "cherry-b300-8x"]
    }
}
```

`mlnode_foundry/runner.py` подбирает runner с учётом affinity + бюджет (через `cue export tools/runners.cue` → JSON → Python).

## 9. Bidirectional flow `mlnode ↔ experiments`

```
   profiles/*.cue         build & publish        Stage 4 image
   ───────────────►  CI  ──────────────────►  GHCR
                                                  │
                          (manual / label /       │ pull
                           cron trigger)          ▼
   benchmark.yml  ───────────────────►   poc-benchmark agent
                                            (vast.ai / ssh)
                                                  │
                                                  │ writes
                                                  ▼
                                       experiments/<YYYY-MM>/<id>/
                                         README.md, nonces.json, metrics.json
                                                  │
                                                  │ auto-PR
                                                  ▼
                                       profiles/*.cue::metrics + status
```

### Контракт `mlnode → experiments`

GHA workflow `benchmark.yml` (workflow_dispatch / PR-метка):

```yaml
on:
  workflow_dispatch:
    inputs:
      profile: { required: true }
      runner:  { required: true }
      tier:    { default: 'gpu-smoke' }     # gpu-smoke | benchmark
```

Workflow резолвит профиль → image digest, runner config → публикует событие в queue (issue/Slack/artifact). **CI не ждёт результата** — бенч может занять 30-90 минут.

### Контракт `experiments → mlnode-foundry`

После завершения Tier 3 агент:

1. Пишет `experiments/<YYYY-MM>/<package-tag>/`:
   - `README.md` — отчёт
   - `nonces.json` — собранные nonces
   - `logprobs.json` — для cross-validation
   - `metrics.json` — `expected_nonces_per_min`, `cost_estimate_usd`
2. Открывает PR в `kaitakuai/mlnode-foundry` обновляющий **только `state/<package-tag>.json`**:
   - `validation.benchmark` блок
   - `metrics:` блок
   - `status: benchmarked`
3. **Профиль `profiles/<x>.cue` НЕ трогается** — это pure spec, агент его не пишет.
4. PR не триггерит Stage 4 ребилд: state-файлы не входят в `profile_hash`-input.

## 10. AI-ассистент `poc-benchmark` интеграция

Контракт агента (см. [.claude/agents/poc-benchmark.md](../../.claude/agents/poc-benchmark.md)):

- **Input**: resolved profile YAML + runner config + Stage 4 image digest
- **Output**: `experiments/<YYYY-MM>/<package-tag>/{README.md, nonces.json, metrics.json}` + auto-PR в mlnode
- **Идемпотентность**: повторный запуск с тем же `(image_digest, runner)` пропускается
- **Failure handling**: агент сам поднимает упавший vast.ai-инстанс или фейлит корректно
- **Стоимость**: `cost_estimate_usd` записывается в каждый `metrics.json`; dashboard агрегирует cumulative spend

### Стартовый режим: local-trigger

Запуск агента вручную через Claude Code на машине разработчика, когда нужен бенч. Минимум инфры, нулевой ramp-up. Headless watcher (отдельный always-on runner с Claude Code) — миграция через 2-3 месяца если ручной trigger станет узким местом.

## 11. Supply chain

| Practice | Реализация |
|----------|------------|
| SLSA L3 provenance | `--attest type=provenance,mode=max` на каждый push |
| SBOM (SPDX) | `--sbom=true` в BuildKit |
| Rendered Dockerfile attestation | `--attest type=dockerfile` (in-toto predicate) — позволяет аудитору cвалидировать «какой Dockerfile собрал данный digest» |
| Cosign keyless signing | GitHub OIDC → `cosign sign --yes` в `promote.yml` |
| Dashboard verify | `cosign verify` с identity-regex по workflow path в `sync_registry.py` |
| Pin-цепочка | Stage 2 (vllm-poc) digest → `tools/stage3.lock.cue`; Stage 3 (mlnode-base) digest → `profiles/_resolved/<id>.json` (генерится в CI через `cue export`) |
| `:latest` запрещён | проверка в `promote.yml` |
| Profile schema check | `cue vet profiles/<x>.cue tools/naming.cue` в `validate-profiles.yml` — type-checking + axes catalog + sum-type discrimination за один вызов |
| Tag-format check | regex синхронный с dashboard, в `validate-profiles.yml` |

## 12. Build optimization

### Промежуточные образы как first-class артефакты

Каждый Stage публикуется в GHCR, переиспользуется через `@sha256:<digest>`. Stage 3 (apt + OpenSSL + poetry + venv) — **строится один раз** на (mlnode, vllm) пару, потом 8+ Stage 4 сборок пользуются результатом через `FROM kaitakuai/mlnode-base@sha256:…`.

### Кеш-уровни

| Уровень | Механизм |
|---------|----------|
| BuildKit cache mounts | `--mount=type=cache` для apt/pip/poetry — наследуем апстрим-настройку |
| Registry-backed cache | `cache-from/to type=registry,ref=:buildcache,mode=max` |
| Smart rebuild detection | `tools/build-hash.py` — content-hash (`profile_hash`); если совпало с лейблом текущего тэга, билд skip-ается |
| Параллелизм | `strategy.matrix` по профилям, `max-parallel: 10+` |

### Retention policy

| Артефакт | Политика |
|----------|----------|
| Stage 1 / Stage 2 теги (residual fork, vllm-poc) | ≥ 6 мес, GC через год |
| Stage 3 теги (mlnode-base) | ≥ 3 мес, GC если нет depending Stage 4 |
| Stage 4 теги | **постоянно** — публичный артефакт |
| `:buildcache` теги | последние 5, GC еженедельно |

### Ожидаемые времена (warm cache)

| Сценарий | Время |
|----------|-------|
| Правка одного профиля | ~1 мин |
| Правка shared hw-patch (затрагивает N профилей) | ~1 мин × N параллельно |
| Bump Stage 3 (каскад на все профили) | ~9 мин |
| Bump vLLM (Stage 2 vllm-poc уже опубликован) | ~9 мин |

## 13. Maintenance multipliers

Все включаются с первого PR, чтобы не тратить dev-time потом:

| Multiplier | Что даёт |
|------------|----------|
| **Profile inheritance (`extends:`)** | новые 10 профилей по семейству — ~15 строк каждый |
| **Renovate bot на `tools/stage3.lock.cue`** | авто-PR при новом upstream-теге; merge кнопкой при зелёном CI |
| **vLLM ENV snapshot в `state/_stage3-vllm-env.json`** | компат-валидация профилей — регрессии «переименовали флаг» ловятся в CI через `cue vet state/_stage3-vllm-env.json state/_schema.cue`, не в проде. Snapshot пишется автоматикой при build Stage 3 (introspect через `python -c "import vllm.envs"`); spec-файл `stage3.lock.cue` остаётся чистым pin-ом |
| **`mlnode-foundry profile new/add-model/add-gpu`** | bulk profile generation из defaults |
| **CODEOWNERS разделяет infra и контент** | researcher PR'ит профили без dev-ревью; dev-ревью только на `_base/`, `tools/naming.cue`, `_schema.cue`, `tools/`, `tools/stage3.lock.cue` |
| **Reusable GHA workflows** | один `build-stage4.yml` параметризованный |
| **`notes:` блок в профиле** | через год — почему стоит этот флаг, ссылка на бенч |
| **Dashboard «stale profiles» view** | показывает «12 профилей собраны 3 мес назад при vllm 0.19, Stage 3 уже 0.20» — кнопка `Rebuild all` |

### Отложенные multipliers (по факту первой боли)

- `mlnode-foundry profile lint` — drift-detection между похожими профилями
- HF Hub model card auto-fetch для defaults
- Dashboard family-tree view
- Auto-bump metrics из experiments в профиль
- HW-patch compatibility metadata + linter

## 14. Раскладка репозитория

```
kaitakuai/mlnode-foundry/
├── cue.mod/                              # Cue module manifest (`module: "github.com/kaitakuai/mlnode-foundry"`)
│   └── module.cue
├── stage3/
│   ├── docker-bake.hcl
│   ├── Dockerfile.patch-and-build
│   └── README.md
├── stage4/
│   ├── docker-bake.hcl                   # matrix через render-bake (читает Cue → JSON)
│   ├── Dockerfile                        # один общий, всё через build-args
│   └── README.md
├── profiles/                             # pure spec (intent), пишут люди
│   ├── _schema.cue                       # #BaseProfile | #OverlayProfile sum type
│   ├── _base/                            # base-комбинируются через `&` unification
│   │                                     #   style: max 2 базы на профиль, базы атомарны
│   ├── _templates/                       # для `mlnode-foundry profile new`
│   ├── h100-qwen.cue
│   ├── h200-qwen.cue
│   ├── b200-qwen.cue
│   ├── b200-kimi-int4.cue
│   ├── b300-qwen.cue
│   ├── b300-kimi-int4.cue
│   └── ...
├── state/                                # observed state, пишет автоматика (CI + agent)
│   ├── _schema.cue                       # #State схема (Cue, human-authored)
│   ├── _stage3-vllm-env.json             # observed: vLLM ENV var snapshot (JSON, machine-written)
│   └── mlnode-<gpu>-<model>-<tag>.json   # один файл на каждый publish (JSON, machine-written)
├── tools/                                # data-only: spec и policy файлы (intent), без Python-кода
│   ├── naming.cue                        # axes registry + naming policy (объединено)
│   ├── runners.cue                       # runner inventory (один файл, не каталог)
│   ├── stage3.lock.cue                   # IMMUTABLE pin: upstream commit + patches list + Stage2 (vllm-poc) digest
│   ├── hw-patches/                       # *.dockerfile фрагменты (по имени)
│   └── runner-patches/                   # *.py патчеры (по имени)
├── mlnode_foundry/                         # Python CLI пакет (Typer); ставится `pip install -e .`
│   ├── __init__.py
│   ├── cli.py                            # Typer entrypoint, subcommand tree
│   ├── cue.py                            # subprocess-обёртка над `cue eval/export/vet`
│   ├── render_bake.py                    # profile + naming.cue → bake build-args + matrix
│   ├── render_name_tag.py                # profile + naming.cue → package-name + tag
│   ├── build_hash.py                     # profile_hash для skip-if-unchanged
│   ├── validate.py                       # вызов `cue vet` + custom checks (style-конвенции)
│   ├── runner.py                         # runner selection из `cue export tools/runners.cue`
│   ├── expand.py                         # self-contained profile snapshot (`cue eval`)
│   └── profile/                          # подкоманды `mlnode-foundry profile ...`
│       ├── __init__.py
│       ├── new.py                        # `profile new --gpu X --model Y [--quant Z]`
│       └── bulk.py                       # `profile add-model`, `profile add-gpu`
├── pyproject.toml                        # объявляет mlnode_foundry пакет + console_script `mlnode-foundry`
├── patches/
│   └── 0001-content-type-middleware.patch
├── docs/
│   ├── architecture.md                   # этот файл
│   ├── adr/                              # 0001..0014
│   ├── decision-log.md
│   └── runbooks/
├── renovate.json                         # auto-PR на upstream pins
├── CODEOWNERS                            # infra vs content split
└── .github/workflows/                    # GHA требует YAML — единственное место с YAML
    ├── validate-profiles.yml             # Tier 0+1
    ├── build-stage3.yml
    ├── build-stage4.yml                  # пишет state/<x>.json после publish
    ├── benchmark.yml                     # Tier 2/3 trigger (out-of-band)
    └── promote.yml                       # cosign sign + sbom + state/ commit
```

**Конфигурационные файлы — Cue.** Cue одновременно язык схем и данных — отдельные `_schema.json` файлы не нужны. Только GHA workflows остаются YAML (внешний контракт), `pyproject.toml` остаётся TOML (Python ecosystem контракт).

**`registry/` каталог удалён.** Метаданные образа доступны через OCI labels + cosign attestations + state-файлы; dashboard читает оттуда напрямую через `crane manifest` и `cosign download attestation`. См. ADR-0011.

**Консолидация спец-файлов** (D7): было 7-8 типов (axes.yaml + naming.yaml + runners/*.yaml + stage2.lock.yaml + 4 schema-файла), стало 4 (`naming.cue` + `runners.cue` + `stage3.lock.cue` + profile/state); схемы embedded в Cue, отдельных schema-файлов нет.

**Cue dependency.** `cue` binary (~15 MB Go-based CLI) ставится через `brew install cue-lang/tap/cue` (macOS), `go install cuelang.org/go/cmd/cue@latest`, или скачиванием из [GitHub Releases](https://github.com/cue-lang/cue/releases). Версия пинится в `pyproject.toml` через `mise.toml` (см. P11 в [ADR-0008](./adr/0008-custom-dsl-vs-frameworks.md)).

Этот layout — для **нового** репо `kaitakuai/mlnode-foundry`. Старый `kaitakuai/mlnode` остаётся со своей текущей структурой (`full/`, `overlay/`, `tools/fragments/`, `tools/generate-dockerfiles.py`) до архивации; никаких удалений в нём не делаем.

`.mlnode-src/` в новом репо НЕ создаётся. Stage 3 потребляет апстрим `gonka-ai/gonka` через `--build-context` напрямую (см. §6 / [ADR-0001](./adr/0001-four-stage-pipeline.md)).

## 15. CLI команды

`mlnode-foundry` — Typer-based Python CLI, единый entrypoint. Устанавливается через `pip install -e .` (dev) или `pip install mlnode-foundry` (когда опубликуем в PyPI). Subcommand tree:

```bash
# Build / inspect / publish image
mlnode-foundry build <profile>                  # local build, no push, no sign
mlnode-foundry dockerfile <profile>             # rendered Dockerfile → stdout
mlnode-foundry tag <profile>                    # computed package:tag
mlnode-foundry hash <profile>                   # computed profile_hash
mlnode-foundry expand <profile>                 # self-contained snapshot
mlnode-foundry publish <profile>                # build + push + sign + state/.json (CI only)
mlnode-foundry benchmark <profile> --runner <name>   # печатает команду для запуска агента

# Profile authoring
mlnode-foundry profile validate <profile>       # `cue vet` + style-конвенции (max 2 base-импорта) + ENV compat
mlnode-foundry profile new --gpu X --model Y [--quant Z]
mlnode-foundry profile add-model <name> --quants fp8,int4   # bulk for all GPUs
mlnode-foundry profile add-gpu <name> --sm 100              # bulk for all models
mlnode-foundry profile lint                                 # drift-detection (deferred multiplier)

# Runner inventory
mlnode-foundry runner list
mlnode-foundry runner select <profile> --tier benchmark     # affinity + budget pick

# Self-help
mlnode-foundry --help
mlnode-foundry profile --help
mlnode-foundry --install-completion             # autocomplete для bash/zsh
```

Все аргументы type-checked (профиль должен реально существовать, runner — быть в `runners.cue`); help auto-generated; subcommand tree расширяется без правки entrypoint-а.

## 16. Чёрный ящик: «дать образ X»

### Минимальный input — один файл

`profiles/<gpu>-<model>[-<quant>].cue`. Всё остальное resolved-ится автоматически:

| Implicit input | Откуда берётся |
|----------------|----------------|
| Stage 3 digest | `tools/stage3.lock.cue` (`kaitakuai-base` mode) или из профиля (`upstream-overlay` mode) |
| `tools/naming.cue` | реестр identity-осей + глобальная политика имени и тэга |
| `profiles/_schema.cue` | sum-type схема `#BaseProfile \| #OverlayProfile` |
| `profiles/_base/<x>.cue` | базы, импортируемые профилем через unification |
| `tools/hw-patches/<name>.dockerfile` | по списку имён в профиле |
| `tools/runner-patches/<name>.py` | по `runner_patch` в профиле |
| `stage4/Dockerfile` | один общий template |
| Cosign keyless OIDC token | в workflow |

### Outputs

```
1. Docker image:
   ghcr.io/kaitakuai/mlnode-b300-kimi:0.2.13-vllm0.20.0-q.int4-k1
   digest: sha256:abc123...

2. Attestations (привязаны к digest, source of truth для метаданных):
   - SLSA L3 provenance — где, когда, чем собрано
   - SBOM (SPDX) — что внутри (пакеты, версии)
   - Rendered Dockerfile (in-toto predicate) — точный Dockerfile, которым собран этот digest
   - Cosign signature (keyless, GitHub OIDC) — кто подписал

3. OCI labels на образе (читаются `crane manifest`):
   - gonka.kaitaku.profile_hash
   - gonka.kaitaku.axes (JSON)
   - gonka.kaitaku.versions (JSON)
   - gonka.mlnode.variant
   - gonka.kaitaku.revision

4. state/<package-tag>.json в репо (observed state, schema в state/_schema.cue):
   - profile reference + profile_hash
   - image digest + built_at + cosign_identity
   - status (draft/validated/benchmarked/deprecated)
   - validation results per tier
   - metrics (заполняется агентом после Tier 3)

5. Build cache в GHCR (:buildcache теги)
```

Метаданные больше не дублируются в `registry/*.json` — этот каталог удалён. Dashboard читает образ + attestations + state-файл напрямую.

### Контракт инвариантов

- Один и тот же профиль + те же resolved inputs ⇒ тот же `profile_hash` ⇒ существующий тэг в GHCR (skip rebuild).
- Изменение **любого** resolved input ⇒ другой `profile_hash` ⇒ новый `rev` или ошибка валидации.
- Образ невозможно опубликовать без подписи — `promote.yml` падает если cosign не отработал.
- Имя пакета и тэг **нельзя** задать руками — они выводятся из профиля + `tools/naming.cue`.
- Профиль не редактируется автоматикой; `state/` не редактируется людьми.

## 17. Гибкость по новым осям

Таблица показывает стоимость добавления новой axis-семантики:

| Сценарий | Сложность | Что меняется |
|----------|-----------|--------------|
| Новая GPU (`l40`, `mi300`) | низкая | один профиль (или `_base/<gpu>.cue` + N профилей) |
| Новая модель (`deepseek`) | низкая | один профиль на каждый GPU; `_base/<model>.cue` если общая логика |
| Новая ось `quant` (FP8 / INT4 / NVFP4) | низкая | строка в `axes:` блоке `tools/naming.cue` + полем в существующих профилях |
| Новая ось `framework` (sglang) | средняя | строка в `axes:` блоке `tools/naming.cue`; Stage 1-3 параметризуются по `framework`; dashboard не меняется (JSONB) |
| Новая ось `transform` (минимизация) | средняя | placeholder уже зарезервирован; добавляется Stage 5 + стратегии (см. §18) |
| Runtime-ось (без нового образа) | нулевая | блок `runtime_defaults` в профиле или override при `docker run` |

## 18. Stage 5 placeholder — image minimization

Ось `transform` зарезервирована в `tools/naming.cue` (`status: reserved`, `allowed_values: [full, slim]`). Блок `minimization:` в профиле зарезервирован в JSON-Schema, но игнорируется build-системой до реализации.

Когда потребуется минимизация:

1. Реализуется `stage5/Dockerfile.minimize` (~30 строк).
2. Создаётся `tools/minimize-strategies/` с реализациями (`multistage-copy.py`, `apko-distroless.py`, `dockerslim.py`).
3. Профили дополняются блоком `minimization:` опционально.
4. Имена/тэги авто-обновляются: `transform: slim` → суффикс `-t.slim` в тэге.
5. Dashboard подхватывает новую ось через `axes_catalog` без миграций.

Чанки supply-chain, валидация и build-optimization применяются к slim-образам автоматически.

## 19. ADR Index

| ADR | Тема | Статус |
|-----|------|--------|
| [0001](./adr/0001-four-stage-pipeline.md) | Five-stage build pipeline | Accepted (amended 2026-06-21) |
| [0002](./adr/0002-tag-and-naming-scheme.md) | Tag and naming scheme | Accepted |
| [0003](./adr/0003-profile-dsl-and-axis-types.md) | Profile DSL and axis types | Accepted |
| [0004](./adr/0004-supply-chain-attestations.md) | Supply-chain attestations | Accepted |
| [0005](./adr/0005-dashboard-jsonb-schema.md) | Dashboard JSONB schema | Accepted |
| [0006](./adr/0006-rendered-dockerfile-policy.md) | Rendered Dockerfile policy | Accepted |
| [0007](./adr/0007-build-optimization-and-caching.md) | Build optimization and caching | Accepted |
| [0008](./adr/0008-custom-dsl-vs-frameworks.md) | Why custom thin DSL over Bazel/apko/Dagger | Accepted |
| [0009](./adr/0009-validation-tiers-and-benchmarks.md) | Validation tiers and benchmark integration | Accepted |
| [0010](./adr/0010-image-minimization-stage4.md) | Image minimization (Stage 5 placeholder) | Reserved |
| [0011](./adr/0011-spec-state-policy-separation.md) | Spec / state / policy разделены: `profiles/` (intent), `state/` (observed), `tools/naming.cue` (org policy); `registry/` каталог удалён (читаем OCI напрямую) | Accepted |
| [0012](./adr/0012-cue-as-config-language.md) | Cue как единый язык для конфигов, схем и validation; sum types, unification, embedded type system | Accepted |
| [0013](./adr/0013-poc-integration-architecture.md) | PoC integration architecture (Stage 1/2 split: residual + gonka-poc) | Accepted |
| [0014](./adr/0014-residual-fork-permanent-infra.md) | Residual vLLM fork as permanent infrastructure | Accepted |

## 20. Стратегия репозитория (greenfield)

Build-система живёт в новом репо `kaitakuai/mlnode-foundry`. Существующий `kaitakuai/mlnode` замораживается с момента создания нового и архивируется через ~1 квартал стабильной работы.

GHCR-неймспейсы не пересекаются:

- Старый репо публикует `ghcr.io/kaitakuai/mlnode-full:*` и `mlnode-overlay:*` — существующие теги остаются доступны навсегда после архивации.
- Новый репо публикует `ghcr.io/kaitakuai/mlnode-base:*` (Stage 3) и `ghcr.io/kaitakuai/mlnode-<gpu>-<model>:*` (Stage 4).
- Stage 1 (`ghcr.io/kaitakuai/vllm-sampler-residual:*`) и Stage 2 (`ghcr.io/kaitakuai/vllm-poc:*`) — общие, в репо `kaitakuai/vllm`.

Греenфилд-подход выбран ради минимизации риска для production (нод-операторы пинят digest-ы из старого репо), параллельной разработки без конфликтов и тривиального отката (продолжаем использовать legacy).

## Источники

- [Upstream Gonka mlnode](https://github.com/gonka-ai/gonka/tree/main/mlnode) — Stage 3 потребляет `mlnode/packages/api/Dockerfile`
- [kaitakuai/vllm](https://github.com/kaitakuai/vllm) — репо Stage 1 (residual) + Stage 2 (vllm-poc)
- [Cue language](https://cuelang.org/)
- [docker buildx bake](https://docs.docker.com/build/bake/)
- [BuildKit attestations](https://docs.docker.com/build/metadata/attestations/)
- [SLSA framework](https://slsa.dev/)
- [Sigstore / cosign](https://www.sigstore.dev/)
- [.claude/agents/poc-benchmark.md](../../.claude/agents/poc-benchmark.md) — контракт AI-агента для Tier 3 валидации
