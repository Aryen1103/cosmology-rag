import type { AgentStep } from '../api'

function formatInput(input: AgentStep['input']): string {
  return Object.entries(input)
    .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
    .join(', ')
}

export function AgentSteps({ steps }: { steps: AgentStep[] }) {
  if (steps.length === 0) return <p className="muted agent-steps">The agent answered without calling any tools.</p>
  return (
    <details className="card agent-steps">
      <summary>
        Agent trace: {steps.length} tool call{steps.length === 1 ? '' : 's'}
      </summary>
      <ol>
        {steps.map((step, i) => (
          <li key={i} className={step.is_error ? 'is-error' : undefined}>
            <code>
              {step.tool}({formatInput(step.input)})
            </code>
            <span className="meta">{step.output}</span>
          </li>
        ))}
      </ol>
    </details>
  )
}
