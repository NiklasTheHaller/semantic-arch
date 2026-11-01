# Agent Engineering Guidelines & Architectural Principles

All code, architecture, and technical decisions in this repository MUST strictly adhere to the following software engineering standards. Because this project is built via **AI-driven agentic development**, we enforce a Cargo workspace of small, isolated crates, narrow trait boundaries, and compiler-enforced invariants to keep agent context windows clean and prevent regressions. Do not take short-term shortcuts that create long-term refactoring debt.

## 1. Test-Driven Development (TDD), Snapshot Testing & Red-Green-Refactor
We practice strict Test-Driven Development wherever applicable:
1. **Red (Write the Test First):** Before writing implementation code, write unit or integration tests that define the desired behavior and edge cases. Verify that the tests fail for the expected reason, and ensure they genuinely test the target functionality rather than tautologies.
2. **Green (Make It Pass):** Write the clean, minimal production code required to make the failing tests pass.
3. **Refactor (Clean Up — Mandatory):** Do not skip the refactoring phase. Once tests are green, elevate the code to state-of-the-art quality: remove duplication, improve abstractions, verify SOLID alignment, and ensure all tests remain green.
4. **Snapshot Testing (`insta` crate) for Complex Outputs & Parsers:** When testing complex serializers, AST transformations, or structured document parsers against real-world fixtures, use `insta::assert_yaml_snapshot!` or `insta::assert_debug_snapshot!` rather than writing brittle, multi-hundred-line manual `assert_eq!` trees that agents can hallucinate or misalign.

## 2. Hexagonal / Clean Architecture & SOLID Principles
- **Hexagonal Architecture (Ports & Adapters) / Clean Architecture:**
  - **The Inside (`core-domain` & `core-application` crates):** Must remain 100% pure and independent of frameworks, databases, file systems, network I/O, UI, and external infrastructure libraries. Dependencies point strictly **inward** toward the Domain Core.
  - **Inbound/Driving Ports:** Defined as explicit Use Case traits or command handlers in the Application layer (e.g., `IngestDocumentUseCase`, `ExecuteQueryUseCase`, `ExportReportUseCase`). Driven by Primary Adapters (`adapter-cli`, `adapter-http`, `adapter-desktop-ui`).
  - **Outbound/Driven Ports:** Defined as narrow, role-specific traits in the Domain/Application core (e.g., `DocumentParserPort`, `ArtifactExportPort`, `EntityRepositoryPort`, `SearchIndexPort`). Implemented by Secondary/Infrastructure Adapters (`adapter-parser-xml`, `adapter-storage-sqlite`, `adapter-search-engine`).
- **Single Responsibility Principle (SRP):** Every crate, module, struct, and function must have one clear responsibility and one reason to change.
- **Open-Closed Principle (OCP):** Design systems so behavior can be extended (e.g., adding new input file dialects, protocol versions, or export formats via new outbound adapter crates) without modifying existing core domain logic.
- **Liskov Substitution Principle (LSP):** Implementations of ports/traits must honor the exact behavioral contracts and invariants expected by callers.
- **Interface Segregation Principle (ISP):** Prefer small, role-specific traits over bloated, general-purpose ones. Split read and write operations into separate traits where appropriate.
- **Dependency Inversion Principle (DIP):** High-level domain policies must not depend on low-level infrastructure details (e.g., specific database engines, file systems, serialization formats, or UI frameworks). Both must depend on well-defined domain abstractions.
- **Balancing CCP and CRP across Cargo Crates:**
  - **Common Closure Principle (CCP):** Group modules that change together for the same reasons into the same crate.
  - **Common Reuse Principle (CRP):** Do not force consumers of a crate to depend on transitive dependencies they do not need. Split crates when downstream consumers only require a subset of functionality.

## 3. Intentional Naming Conventions
- **Reveal Intent, Not Implementation:** Names must express *what* a concept represents in the domain or *why* it exists—never *how* it is implemented under the hood (e.g., use `ExternalSystemIdentifier` or `ValidatedPayload` rather than `RegexExtractedStringMap` or `RawByteVec`).
- **Ubiquitous Language:** Establish and enforce a strict, consistent domain vocabulary across all crates, types, traits, and module names. Never use synonymous terms for the same domain concept across different layers.

## 4. Small, Pure Functions
- **Do One Thing Well:** Keep functions small, focused, and operating at a single level of abstraction.
- **Purity First:** Separate pure domain logic and data transformations from side effects (I/O, filesystem access, network calls, mutable state). Pure functions are deterministic, easier to reason about in isolated context windows, and trivially testable without mocks.

## 5. Self-Documenting Code & "Why-Only" Comments
- **What vs. Why:** Never write comments that explain *what* the code is doing. If the code requires a comment to explain *what* it does, refactor the code and improve the variable, type, or function names until it is self-documenting.
- **Documenting Rationale ("Why"):** Comments must **only** explain *why* something exists, why a specific threshold, heuristic, or workaround was chosen, or why an alternative approach was rejected. Whenever an important architectural or edge-case decision is made, document the rationale directly in the code so future maintainers and agents understand the context.

## 6. Strict Type Safety, Postel's Law & Workspace Lints
- **No Untyped Escape Hatches:** Never use `any` (if using TypeScript in a frontend adapter) or untyped dynamic maps (`serde_json::Value` inside core domain logic). External untyped payloads must be immediately validated and parsed at the adapter boundary into strict domain types ("Parse, don't validate").
- **Make Illegal States Unrepresentable + Postel's Law ("Strict Types, Tolerant Ingestion"):** Leverage the type system (Rust algebraic data types, exhaustive `enum` matching, and validated newtypes) to enforce domain invariants at compile time. When ingesting messy external or legacy data containing non-fatal anomalies (e.g., minor rounding discrepancies or deprecated optional fields), do not crash whole-file ingestion; represent the anomaly explicitly in the domain type (e.g., `ValidationStatus::MinorDiscrepancy { delta }`) so the caller or user can inspect and resolve it.
- **Explicit Error Handling:** Use `thiserror` for structured, domain-specific error enums in library/core crates so callers can match exhaustively on failure modes. Reserve opaque error wrappers for top-level binary entrypoints.
- **Centralized Workspace Lints (`Cargo.toml`):** Every crate in the workspace must inherit `[workspace.lints]` enforcing:
  - `unsafe_code = "forbid"`
  - `clippy::unwrap_used = "deny"`
  - `clippy::expect_used = "deny"` (outside of `#[cfg(test)]` modules)
  - `clippy::float_arithmetic = "deny"` (where exact precision is required—such as monetary, accounting, or physical quantity math—enforce fixed-point `rust_decimal::Decimal` or integer base units).