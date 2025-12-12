# Architecture Diagram: --save_intermediates Feature

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER COMMAND                               │
├─────────────────────────────────────────────────────────────────────┤
│  wisq circuit.qasm --mode full_ft -si ./intermediates                │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ARGUMENT PARSER                                   │
│  (src/wisq/__init__.py: main())                                      │
├─────────────────────────────────────────────────────────────────────┤
│  • Parses --save_intermediates flag                                  │
│  • Sets args.save_intermediates = "./intermediates"                  │
│  • Routes to appropriate compilation mode                           │
└──────────┬──────────────────────────────┬──────────────────────────┘
           │                              │
     FULL_FT_MODE               (Other modes: OPT_MODE, SCMR_MODE)
           │                              │
           ▼                              │
┌──────────────────────────────────────┐  │
│  compile_fault_tolerant()            │  │
│  (save_intermediates_dir parameter)  │  │
└──────┬───────────────────────────────┘  │
       │                                   │
       ▼                                   │
┌──────────────────────────────────────┐  │
│ Check for existing intermediates     │  │
│ has_intermediate_file(dir)           │  │
└──────┬───────────────────────────────┘  │
       │                                   │
    ┌──┴──┐                                │
    │     │                                │
   YES   NO                                │
    │     │                                │
    ▼     ▼                                │
  SKIP  RUN                                │
  OPT   OPT                                │
    │     │                                │
    │  SAVE                                │
    │  save_intermediate_file()            │
    │     │                                │
    └──┬──┘                                │
       │                                   │
       ▼                                   │
┌──────────────────────────────────────┐  │
│  MAP & ROUTE                         │  │
│  map_and_route()                     │  │
│  (save_intermediates_dir parameter)  │  │
└──────┬───────────────────────────────┘  │
       │                                   │
       └──────────┬──────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  OUTPUT RESULT │
         │   (JSON file)  │
         └────────────────┘
```

## Class/Function Hierarchy

```
main()
├── Argument Parser
│   └── --save_intermediates / -si flag
│
├── If FULL_FT_MODE:
│   └── compile_fault_tolerant(save_intermediates_dir)
│       │
│       ├── has_intermediate_file(save_intermediates_dir)
│       │   ├── Check file existence
│       │   └── Return boolean
│       │
│       ├── If intermediate exists:
│       │   └── Load: get_intermediate_filepath()
│       │
│       ├── If not exists:
│       │   ├── optimize() - decompose & optimize
│       │   └── save_intermediate_file() - save result
│       │       └── os.makedirs() - create dir if needed
│       │
│       └── map_and_route(save_intermediates_dir)
│
└── If SCMR_MODE:
    └── map_and_route(save_intermediates_dir)
```

## State Machine

```
                    ┌──────────────────┐
                    │   Start: Run 1   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Check for saved? │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐ NO
                    │  File exists?    ├──────┐
                    └────────┬─────────┘      │
                            YES              │
                             │               │
                    ┌────────▼──────────┐ ┌──▼──────────────┐
                    │ Load saved QASM   │ │ Run optimize()  │
                    │ Skip optimization │ │ Save result     │
                    └────────┬──────────┘ └──┬──────────────┘
                             │                │
                             └────────┬───────┘
                                      │
                    ┌─────────────────▼────────────┐
                    │  Run map_and_route()         │
                    │  (Same for all subsequent)   │
                    └─────────────────┬────────────┘
                                      │
                    ┌─────────────────▼────────────┐
                    │  Output JSON result          │
                    └──────────────────────────────┘
