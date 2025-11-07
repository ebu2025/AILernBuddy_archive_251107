# Process Engine & BPMN Modeling Framework

This guide documents the BPMN assets and analysis utilities that support
adaptive learning journeys.  The artifacts emphasise clear process scope,
explicit triggers, and measurable end states so that orchestration tooling
can align interventions across sequential and parallel skill development.

## Core Modeling Principles

- **Scope clarity** – each start and end event includes documentation that
  states the journey boundaries, the trigger condition, and the exit
  criteria learners must satisfy.
- **Left-to-right flow** – diagrams are organised to reduce cross-overs and
  keep narrative momentum, making the models easier to review in discovery
  workshops.
- **Externalised rules** – gateways are annotated with simple expressions
  (e.g., `diagnostic_score &lt; 0.6`) that can be implemented as service tasks
  or decision tables outside of the BPMN diagram.
- **Stakeholder-specific views** – every skill area now ships with a
  summary and detailed diagram so that programme leads, facilitators, and
  analysts can inspect the level of fidelity they require.

## Learning Journey Templates

| Skill Area | Summary Diagram | Detailed Diagram | Structuring Notes |
| --- | --- | --- | --- |
| Foundational Literacy | `process_models/foundational_literacy.bpmn` (`Process_FoundationalLiteracy_Summary`) | `process_models/foundational_literacy.bpmn` (`Process_FoundationalLiteracy_Detailed`) | Sequential placement with optional bridge modules before dual-track phonics/comprehension work. |
| Data Fluency | `process_models/data_fluency.bpmn` (`Process_DataFluency_Summary`) | `process_models/data_fluency.bpmn` (`Process_DataFluency_Detailed`) | Parallel descriptive/visualisation sprint with inclusive focus-track selection to support personalised emphasis. |
| Collaborative Capstone | `process_models/collaborative_capstone.bpmn` (`Process_CollabCapstone_Summary`) | `process_models/collaborative_capstone.bpmn` (`Process_CollabCapstone_Detailed`) | Dual workstreams with an embedded risk escalation timer event and iterative go/no-go loop prior to demo approval. |

Each template captures:

- **Start Event Trigger** – e.g., "Trigger: Learner Intake" or "Trigger:
  Personalised Data Fluency Plan".
- **Key Milestones** – major tasks and sprints grouped via exclusive,
  parallel, or inclusive gateways to illustrate sequential versus
  concurrent development.
- **Exit State** – explicit conditions such as "End: Mastery Badge Issued"
  or "End: Demo Approved" so downstream automations know when to launch the
  next journey.

### Alternative Visualisations

- **Summary diagrams** focus on the core milestones and minimal gateway
  logic.  They are optimised for executive briefings or programme design
  reviews.
- **Detailed diagrams** add conditional branching, remediation loops, and
  risk events used by facilitators and the learning operations team.
- To publish stakeholder-specific views, export the relevant process from
  the BPMN file and embed it into miro/Confluence canvases or generate SVG
  snapshots using `bpmn-js` tooling.

## Process Mining Integration

A lightweight analytics module lives in `process_models/process_mining.py`.
It ingests raw journey events and surfaces diagnostics that inform process
improvement experiments:

```python
from process_models.process_mining import parse_event_log, generate_process_diagnostics

events = parse_event_log(event_payloads)
diagnostics = generate_process_diagnostics(
    events,
    end_activity="Capstone",
    top_n_bottlenecks=5,
)
```

Key outputs include cycle-time statistics, variant frequencies, detected
bottlenecks, dropout cases, and skill-area throughput.  The functions avoid
third-party dependencies so they can run inside notebooks, tests, or FastAPI
routes.  The accompanying unit tests (`tests/test_process_mining.py`) use
representative logs to ensure the diagnostics remain stable as additional
metrics are layered on.

## Experimentation Workflow

1. Pick the appropriate BPMN template (summary or detailed) for the skill
   area under review.
2. Export learning activity events to JSON and run them through the process
   mining helpers.
3. Use bottleneck and dropout insights to adjust gateway conditions or task
   assignments in the BPMN models.
4. Repeat modelling iterations and publish new snapshots for stakeholder
   review.
