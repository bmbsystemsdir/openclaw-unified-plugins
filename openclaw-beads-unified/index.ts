/**
 * OpenClaw Beads Plugin v2 (Unified) - Task Tracking Integration
 *
 * Merged implementation combining features from both Blitz and Dexter plugins.
 *
 * Features:
 * - Auto-detect tasks from conversations (LLM + keyword fallback)
 * - Auto-close tasks when completion is detected
 * - Recall relevant open tasks before responses
 * - Link task events to Graphiti episodes (optional)
 * - Configurable capture filter (user_only option)
 * - CLI integration with `bd` commands
 *
 * Tools:
 * - beads_list: List open tasks
 * - beads_create: Create a new task
 * - beads_close: Close a task with resolution
 * - beads_update: Update task status/priority
 * - beads_show: Show task details
 * - beads_search: Search tasks by keyword
 */

import { Type } from "@sinclair/typebox";
import { exec } from "child_process";
import { promisify } from "util";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

const execAsync = promisify(exec);

// ============================================================================
// Types
// ============================================================================

type CaptureFilter = "all" | "user_only";
type TaskStatus = "open" | "in_progress" | "blocked" | "closed";
type TaskType = "task" | "issue" | "blocker";

interface BeadsConfig {
  workspace: string;
  autoDetect: boolean;
  autoClose: boolean;
  autoCaptureFilter: CaptureFilter;
  detectionThreshold: number;
  defaultPriority: number;
  requireConfirmation: boolean;
  linkToGraphiti: boolean;
  graphitiEndpoint?: string;
  graphitiGroupId?: string;
}

interface Task {
  id: string;
  title: string;
  type: TaskType;
  status: TaskStatus;
  priority: number;
  description?: string;
  estimate?: string;
  created_at?: string;
  updated_at?: string;
  closed_at?: string;
  resolution?: string;
  tags?: string[];
  blockers?: string[];
}

interface TaskDetectionResult {
  hasTasks: boolean;
  tasks: Array<{
    title: string;
    description?: string;
    priority: number;
    type: TaskType;
  }>;
  completedTasks: Array<{
    pattern: string;
    summary: string;
  }>;
  confidence: number;
}

// ============================================================================
// Beads CLI Client (Enhanced)
// ============================================================================

class BeadsClient {
  constructor(
    private readonly workspace: string,
    private readonly logger: { info: (msg: string) => void; warn: (msg: string) => void; error: (msg: string) => void },
  ) {}

  /**
   * Execute a bd command and return raw output
   */
  async exec(args: string): Promise<string> {
    try {
      const { stdout, stderr } = await execAsync(`bd ${args}`, {
        cwd: this.workspace,
        timeout: 30000,
        maxBuffer: 1024 * 1024,
        env: { ...process.env, NO_COLOR: '1' },
      });
      if (stderr && !stdout && !stderr.includes('warning')) {
        throw new Error(stderr);
      }
      return stdout.trim();
    } catch (err: unknown) {
      const error = err as { stderr?: string; message?: string };
      this.logger.error(`bd command failed: ${error.stderr || error.message}`);
      throw err;
    }
  }

  /**
   * Execute a bd command and parse JSON output
   */
  async execJson<T>(args: string): Promise<T> {
    const output = await this.exec(`${args} --json`);
    if (!output) return [] as unknown as T;
    
    // Handle JSONL output (one JSON per line)
    const lines = output.split('\n').filter(line => line.trim());
    if (lines.length === 1) {
      try {
        return JSON.parse(lines[0]);
      } catch {
        return [] as unknown as T;
      }
    }
    
    // Multiple lines = JSONL
    const results: unknown[] = [];
    for (const line of lines) {
      try {
        results.push(JSON.parse(line));
      } catch {
        // Skip non-JSON lines
      }
    }
    return results as unknown as T;
  }

  async list(status?: TaskStatus): Promise<Task[]> {
    try {
      const statusArg = status ? `--status ${status}` : '';
      return await this.execJson<Task[]>(`list ${statusArg}`);
    } catch {
      // Fallback to ready command for open tasks
      try {
        const output = await this.exec('ready');
        const tasks: Task[] = [];
        const lines = output.split('\n');
        for (const line of lines) {
          const match = line.match(/\[(\w+-\w+)\]\s+(.+?)\s+\(P(\d)/);
          if (match) {
            tasks.push({
              id: match[1],
              title: match[2],
              type: 'task',
              status: 'open',
              priority: parseInt(match[3], 10),
            });
          }
        }
        return tasks;
      } catch {
        return [];
      }
    }
  }

  async show(id: string): Promise<Task | null> {
    try {
      return await this.execJson<Task>(`show ${id}`);
    } catch {
      return null;
    }
  }

  async create(title: string, options: { type?: TaskType; priority?: number; description?: string } = {}): Promise<string> {
    const type = options.type || 'task';
    const priority = options.priority || 2;
    let cmd = `create "${title.replace(/"/g, '\\"')}" -t ${type} -p ${priority}`;
    
    if (options.description) {
      cmd += ` --description "${options.description.replace(/"/g, '\\"')}"`;
    }
    
    const output = await this.exec(cmd);
    
    // Extract ID from output like "✓ Created issue: clawd-abc" or "✅ Created task: **clawd-abc**"
    const match = output.match(/[:\*\s]+(\w+-\w+)/);
    return match ? match[1] : 'unknown';
  }

  async close(id: string, resolution?: string): Promise<void> {
    let cmd = `close ${id}`;
    if (resolution) {
      cmd += ` -r "${resolution.replace(/"/g, '\\"')}"`;
    }
    await this.exec(cmd);
  }

  async update(id: string, options: { status?: TaskStatus; priority?: number }): Promise<void> {
    let cmd = `update ${id}`;
    if (options.status) {
      cmd += ` --status=${options.status}`;
    }
    if (options.priority) {
      cmd += ` --priority=${options.priority}`;
    }
    await this.exec(cmd);
  }

  async search(query: string): Promise<Task[]> {
    try {
      return await this.execJson<Task[]>(`search "${query.replace(/"/g, '\\"')}"`);
    } catch {
      // Fallback: list all and filter
      const all = await this.list();
      const lower = query.toLowerCase();
      return all.filter(t => 
        t.title.toLowerCase().includes(lower) ||
        t.description?.toLowerCase().includes(lower) ||
        t.id.toLowerCase().includes(lower)
      );
    }
  }

  async sync(): Promise<void> {
    await this.exec('sync');
  }

  async ready(): Promise<Task[]> {
    try {
      return await this.execJson<Task[]>('ready');
    } catch {
      return this.list('open');
    }
  }
}

// ============================================================================
// Task Detector (Enhanced)
// ============================================================================

class TaskDetector {
  constructor(
    private readonly llmModel: string,
    private readonly logger: { info: (msg: string) => void; warn: (msg: string) => void },
  ) {}

  async detect(conversation: string): Promise<TaskDetectionResult> {
    const systemPrompt = `You are a task detector. Analyze conversations to identify:
1. New tasks that should be created (action items, TODOs, requests)
2. Completed tasks being reported

Task indicators:
- "I need to...", "Can you...", "Please...", "TODO:", "We should..."
- "Remind me to...", "Don't forget to...", "Make sure to..."
- Explicit requests or action items
- Problems that need solving (type: issue)
- Critical blockers (type: blocker)

Completion indicators:
- "Done", "Finished", "Completed", "Fixed", "Resolved"
- "I've updated...", "I've created...", "I've fixed..."
- Reports of work being finished

Priority guide:
- 1 = Urgent/blocking
- 2 = Normal (default)
- 3 = Low priority
- 4 = Nice to have

Respond with JSON only:
{
  "hasTasks": boolean,
  "tasks": [
    {
      "title": "Brief task title (max 80 chars)",
      "description": "Optional longer description",
      "priority": 1-4,
      "type": "task" | "issue" | "blocker"
    }
  ],
  "completedTasks": [
    {
      "pattern": "keyword or ID to match existing task",
      "summary": "what was done"
    }
  ],
  "confidence": 0.0-1.0
}

Be conservative - only identify clear, actionable tasks. Skip vague mentions or hypotheticals.`;

    try {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) return this.fallbackDetect(conversation);

      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: this.llmModel,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: `Analyze:\n\n${conversation}` },
          ],
          temperature: 0.1,
          max_tokens: 500,
        }),
      });

      if (!response.ok) throw new Error(`API error: ${response.status}`);

      const json = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
      const content = json.choices?.[0]?.message?.content;
      if (!content) throw new Error('No content');

      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error('No JSON');

      return JSON.parse(jsonMatch[0]) as TaskDetectionResult;
    } catch (err) {
      this.logger.warn(`Task detection failed: ${String(err)}, using fallback`);
      return this.fallbackDetect(conversation);
    }
  }

  private fallbackDetect(conversation: string): TaskDetectionResult {
    const lower = conversation.toLowerCase();
    const tasks: TaskDetectionResult['tasks'] = [];
    const completedTasks: TaskDetectionResult['completedTasks'] = [];

    // Detect new tasks
    const taskPatterns = [
      /(?:i need to|we need to|please|can you|could you|todo:?)\s+(.{10,80})/gi,
      /(?:remind me to|don't forget to|make sure to)\s+(.{10,80})/gi,
      /(?:we should|you should|let's)\s+(.{10,80})/gi,
    ];

    for (const pattern of taskPatterns) {
      let match;
      while ((match = pattern.exec(conversation)) !== null) {
        const title = match[1].trim().replace(/[.!?]$/, '');
        if (title.length > 10 && title.length < 100) {
          // Avoid duplicates
          if (!tasks.some(t => t.title.toLowerCase() === title.toLowerCase())) {
            tasks.push({
              title: title.slice(0, 80),
              priority: 2,
              type: 'task',
            });
          }
        }
      }
    }

    // Detect completions
    const completionPatterns = [
      /(?:done|finished|completed|fixed|resolved)[.:!]?\s*(.{0,80})?/gi,
      /(?:i've|i have)\s+(?:updated|created|fixed|finished|completed)\s+(.{10,80})/gi,
      /✅\s*(?:closed|completed|done)[:\s]*(.{0,80})?/gi,
    ];

    for (const pattern of completionPatterns) {
      let match;
      while ((match = pattern.exec(conversation)) !== null) {
        completedTasks.push({
          pattern: match[1]?.trim().slice(0, 50) || 'task',
          summary: match[0].trim().slice(0, 100),
        });
      }
    }

    return {
      hasTasks: tasks.length > 0 || completedTasks.length > 0,
      tasks,
      completedTasks,
      confidence: tasks.length > 0 || completedTasks.length > 0 ? 0.5 : 0.1,
    };
  }
}

// ============================================================================
// Graphiti Linker (Optional Integration)
// ============================================================================

class GraphitiLinker {
  constructor(
    private readonly endpoint: string,
    private readonly groupId: string,
    private readonly logger: { info: (msg: string) => void; warn: (msg: string) => void },
  ) {}

  async linkTaskEvent(event: 'created' | 'closed' | 'updated', task: Task, details?: string): Promise<void> {
    try {
      const content = event === 'created'
        ? `Task created: [${task.id}] ${task.title} (P${task.priority}, ${task.type})`
        : event === 'closed'
        ? `Task closed: [${task.id}] ${task.title}${details ? ` — ${details}` : ''}`
        : `Task updated: [${task.id}] ${task.title}${details ? ` — ${details}` : ''}`;

      const response = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: Date.now(),
          method: 'tools/add_memory',
          params: {
            group_id: this.groupId,
            name: `${event}: ${task.id}`,
            episode_body: content,
            source: 'text',
            source_description: `beads ${event} event`,
          },
        }),
      });

      if (response.ok) {
        this.logger.info(`Linked ${event} event for ${task.id} to Graphiti`);
      }
    } catch (err) {
      this.logger.warn(`Failed to link task event to Graphiti: ${String(err)}`);
    }
  }
}

// ============================================================================
// Config Parser
// ============================================================================

const beadsConfigSchema = {
  parse(value: unknown): BeadsConfig {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return {
        workspace: process.env.HOME ? `${process.env.HOME}/clawd` : './clawd',
        autoDetect: true,
        autoClose: true,
        autoCaptureFilter: 'all',
        detectionThreshold: 0.7,
        defaultPriority: 2,
        requireConfirmation: false,
        linkToGraphiti: true,
        graphitiEndpoint: 'http://localhost:8000/mcp/',
        graphitiGroupId: 'default',
      };
    }

    const cfg = value as Record<string, unknown>;

    return {
      workspace: typeof cfg.workspace === 'string' ? cfg.workspace : './clawd',
      autoDetect: cfg.autoDetect !== false,
      autoClose: cfg.autoClose !== false,
      autoCaptureFilter: cfg.autoCaptureFilter === 'user_only' ? 'user_only' : 'all',
      detectionThreshold: typeof cfg.detectionThreshold === 'number' ? cfg.detectionThreshold : 0.7,
      defaultPriority: typeof cfg.defaultPriority === 'number' ? cfg.defaultPriority : 2,
      requireConfirmation: cfg.requireConfirmation === true,
      linkToGraphiti: cfg.linkToGraphiti !== false,
      graphitiEndpoint: typeof cfg.graphitiEndpoint === 'string' ? cfg.graphitiEndpoint : 'http://localhost:8000/mcp/',
      graphitiGroupId: typeof cfg.graphitiGroupId === 'string' ? cfg.graphitiGroupId : 'default',
    };
  },
};

// ============================================================================
// Helpers
// ============================================================================

function formatTask(task: Task): string {
  const statusIcon = task.status === 'in_progress' ? '🔄' 
    : task.status === 'blocked' ? '🚫'
    : task.status === 'closed' ? '✅' 
    : '📋';
  const typeIcon = task.type === 'blocker' ? '🔴' : task.type === 'issue' ? '🟡' : '';
  return `${statusIcon}${typeIcon} [${task.id}] ${task.title} (P${task.priority})`;
}

function formatTaskDetails(task: Task): string {
  const lines = [
    `**${task.title}**`,
    `ID: ${task.id} | Type: ${task.type} | Priority: P${task.priority} | Status: ${task.status}`,
  ];
  if (task.description) lines.push(`\n${task.description}`);
  if (task.estimate) lines.push(`Estimate: ${task.estimate}`);
  if (task.resolution) lines.push(`Resolution: ${task.resolution}`);
  if (task.tags?.length) lines.push(`Tags: ${task.tags.join(', ')}`);
  return lines.join('\n');
}

// ============================================================================
// Plugin Definition
// ============================================================================

const beadsUnifiedPlugin = {
  id: 'openclaw-beads-unified',
  name: 'Beads Task Tracker (Unified)',
  description: 'Unified task detection and management with Graphiti integration',
  kind: 'extension' as const,
  configSchema: beadsConfigSchema,

  register(api: OpenClawPluginApi) {
    const cfg = beadsConfigSchema.parse(api.pluginConfig);
    const client = new BeadsClient(cfg.workspace, api.logger);
    const detector = new TaskDetector('gpt-4.1-nano', api.logger);
    const graphitiLinker = cfg.linkToGraphiti 
      ? new GraphitiLinker(cfg.graphitiEndpoint!, cfg.graphitiGroupId!, api.logger)
      : null;

    api.logger.info(
      `openclaw-beads-unified: registered (workspace: ${cfg.workspace}, autoDetect: ${cfg.autoDetect}, autoClose: ${cfg.autoClose}, captureFilter: ${cfg.autoCaptureFilter}, graphiti: ${cfg.linkToGraphiti})`,
    );

    // ========================================================================
    // Tools
    // ========================================================================

    api.registerTool(
      {
        name: 'beads_list',
        label: 'List Tasks',
        description: 'List open tasks from Beads. Use to see current work items and priorities.',
        parameters: Type.Object({
          status: Type.Optional(Type.Union([
            Type.Literal('open'),
            Type.Literal('in_progress'),
            Type.Literal('blocked'),
            Type.Literal('closed'),
            Type.Literal('all'),
          ], { description: 'Filter by status (default: open)' })),
        }),
        async execute(_toolCallId, params) {
          const { status = 'open' } = params as { status?: TaskStatus | 'all' };

          try {
            const tasks = status === 'all' 
              ? [...await client.list('open'), ...await client.list('in_progress'), ...await client.list('closed')]
              : await client.list(status as TaskStatus);

            if (tasks.length === 0) {
              return { content: [{ type: 'text', text: `No ${status} tasks found.` }], details: { count: 0, status } };
            }

            const lines = tasks.map(formatTask);
            return { 
              content: [{ type: 'text', text: `**${tasks.length} ${status} task(s):**\n\n${lines.join('\n')}` }], 
              details: { count: tasks.length, status, tasks } 
            };
          } catch (err) {
            return { content: [{ type: 'text', text: `Failed to list tasks: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'beads_list' },
    );

    api.registerTool(
      {
        name: 'beads_create',
        label: 'Create Task',
        description: 'Create a new task in Beads.',
        parameters: Type.Object({
          title: Type.String({ description: 'Task title' }),
          type: Type.Optional(Type.Union([
            Type.Literal('task'),
            Type.Literal('issue'),
            Type.Literal('blocker'),
          ], { description: 'Task type (default: task)' })),
          priority: Type.Optional(Type.Number({ description: 'Priority 1-4 (1=urgent, default: 2)' })),
          description: Type.Optional(Type.String({ description: 'Task description' })),
        }),
        async execute(_toolCallId, params) {
          const { title, type = 'task', priority = cfg.defaultPriority, description } = params as {
            title: string; type?: TaskType; priority?: number; description?: string;
          };

          try {
            const id = await client.create(title, { type, priority, description });
            
            // Link to Graphiti
            if (graphitiLinker) {
              await graphitiLinker.linkTaskEvent('created', { id, title, type, priority, status: 'open' });
            }

            return { 
              content: [{ type: 'text', text: `✅ Created ${type}: **${id}** — "${title}" (P${priority})` }], 
              details: { id, title, type, priority } 
            };
          } catch (err) {
            return { content: [{ type: 'text', text: `Failed to create task: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'beads_create' },
    );

    api.registerTool(
      {
        name: 'beads_close',
        label: 'Close Task',
        description: 'Close/complete a task in Beads.',
        parameters: Type.Object({
          id: Type.String({ description: 'Task ID (e.g., clawd-abc)' }),
          resolution: Type.Optional(Type.String({ description: 'Resolution summary' })),
        }),
        async execute(_toolCallId, params) {
          const { id, resolution } = params as { id: string; resolution?: string };

          try {
            // Get task details before closing
            const task = await client.show(id);
            
            await client.close(id, resolution);
            
            // Link to Graphiti
            if (graphitiLinker && task) {
              await graphitiLinker.linkTaskEvent('closed', { ...task, status: 'closed', resolution }, resolution);
            }

            return { 
              content: [{ type: 'text', text: `✅ Closed task: **${id}**${resolution ? ` — ${resolution}` : ''}` }], 
              details: { id, resolution } 
            };
          } catch (err) {
            return { content: [{ type: 'text', text: `Failed to close task: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'beads_close' },
    );

    api.registerTool(
      {
        name: 'beads_update',
        label: 'Update Task',
        description: 'Update task status or priority.',
        parameters: Type.Object({
          id: Type.String({ description: 'Task ID' }),
          status: Type.Optional(Type.Union([
            Type.Literal('open'),
            Type.Literal('in_progress'),
            Type.Literal('blocked'),
          ], { description: 'New status' })),
          priority: Type.Optional(Type.Number({ description: 'New priority (1-4)' })),
        }),
        async execute(_toolCallId, params) {
          const { id, status, priority } = params as { id: string; status?: TaskStatus; priority?: number };

          try {
            await client.update(id, { status, priority });

            const changes = [];
            if (status) changes.push(`status → ${status}`);
            if (priority) changes.push(`priority → P${priority}`);
            
            // Link to Graphiti
            if (graphitiLinker) {
              const task = await client.show(id);
              if (task) {
                await graphitiLinker.linkTaskEvent('updated', task, changes.join(', '));
              }
            }

            return { 
              content: [{ type: 'text', text: `✅ Updated **${id}**: ${changes.join(', ')}` }], 
              details: { id, status, priority } 
            };
          } catch (err) {
            return { content: [{ type: 'text', text: `Failed to update task: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'beads_update' },
    );

    api.registerTool(
      {
        name: 'beads_show',
        label: 'Show Task',
        description: 'Show detailed information about a specific task.',
        parameters: Type.Object({
          id: Type.String({ description: 'Task ID' }),
        }),
        async execute(_toolCallId, params) {
          const { id } = params as { id: string };

          try {
            const task = await client.show(id);
            if (!task) {
              return { content: [{ type: 'text', text: `Task ${id} not found.` }], details: { found: false } };
            }

            return { 
              content: [{ type: 'text', text: formatTaskDetails(task) }], 
              details: { task } 
            };
          } catch (err) {
            return { content: [{ type: 'text', text: `Failed to show task: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'beads_show' },
    );

    api.registerTool(
      {
        name: 'beads_search',
        label: 'Search Tasks',
        description: 'Search tasks by keyword.',
        parameters: Type.Object({
          query: Type.String({ description: 'Search query' }),
        }),
        async execute(_toolCallId, params) {
          const { query } = params as { query: string };

          try {
            const tasks = await client.search(query);
            if (tasks.length === 0) {
              return { content: [{ type: 'text', text: `No tasks matching "${query}".` }], details: { count: 0, query } };
            }

            const lines = tasks.map(formatTask);
            return { 
              content: [{ type: 'text', text: `**${tasks.length} task(s) matching "${query}":**\n\n${lines.join('\n')}` }], 
              details: { count: tasks.length, query, tasks } 
            };
          } catch (err) {
            return { content: [{ type: 'text', text: `Search failed: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'beads_search' },
    );

    // ========================================================================
    // CLI Commands
    // ========================================================================

    api.registerCli(
      ({ program }) => {
        const beads = program
          .command('beads')
          .description('Beads task tracking commands');

        beads
          .command('list')
          .description('List tasks')
          .option('--status <status>', 'Filter by status', 'open')
          .action(async (opts: { status: string }) => {
            try {
              const tasks = await client.list(opts.status as TaskStatus);
              if (tasks.length === 0) {
                console.log(`No ${opts.status} tasks.`);
                return;
              }
              for (const task of tasks) {
                console.log(formatTask(task));
              }
            } catch (err) {
              console.error(`Failed: ${String(err)}`);
            }
          });

        beads
          .command('status')
          .description('Show Beads status')
          .action(async () => {
            try {
              const open = await client.list('open');
              const inProgress = await client.list('in_progress');
              const blocked = await client.list('blocked');
              console.log(`Workspace: ${cfg.workspace}`);
              console.log(`Open: ${open.length} | In Progress: ${inProgress.length} | Blocked: ${blocked.length}`);
              console.log(`Auto-detect: ${cfg.autoDetect} | Auto-close: ${cfg.autoClose}`);
              console.log(`Capture filter: ${cfg.autoCaptureFilter}`);
              console.log(`Graphiti link: ${cfg.linkToGraphiti}`);
            } catch (err) {
              console.error(`Failed: ${String(err)}`);
            }
          });

        beads
          .command('ready')
          .description('Show ready work items')
          .action(async () => {
            try {
              const tasks = await client.ready();
              if (tasks.length === 0) {
                console.log('No ready tasks.');
                return;
              }
              for (const task of tasks) {
                console.log(formatTask(task));
              }
            } catch (err) {
              console.error(`Failed: ${String(err)}`);
            }
          });
      },
      { commands: ['beads'] },
    );

    // ========================================================================
    // Lifecycle Hooks
    // ========================================================================

    // Auto-recall: inject open tasks before response
    api.on('before_agent_start', async (event) => {
      if (!event.prompt || event.prompt.length < 5) return;

      try {
        const openTasks = await client.list('open');
        const inProgressTasks = await client.list('in_progress');
        const blockedTasks = await client.list('blocked');

        const allTasks = [
          ...blockedTasks.map(t => ({ ...t, _sort: 0 })), // Blocked first
          ...inProgressTasks.map(t => ({ ...t, _sort: 1 })), // Then in-progress
          ...openTasks.map(t => ({ ...t, _sort: 2 })), // Then open
        ]
          .sort((a, b) => a._sort - b._sort || a.priority - b.priority)
          .slice(0, 10);

        if (allTasks.length === 0) return;

        const taskLines = allTasks.map(formatTask);

        api.logger.info(`openclaw-beads-unified: injecting ${allTasks.length} tasks into context`);

        return {
          systemContext: `<open-tasks>\nCurrent work items:\n${taskLines.join('\n')}\n</open-tasks>`,
        };
      } catch (err) {
        api.logger.warn(`openclaw-beads-unified: recall failed: ${String(err)}`);
      }
    });

    // Auto-detect and auto-close
    if (cfg.autoDetect || cfg.autoClose) {
      api.on('agent_end', async (event) => {
        if (!event.success || !event.messages || event.messages.length === 0) return;

        try {
          const recentMessages = event.messages.slice(-6);
          const conversationParts: string[] = [];

          for (const msg of recentMessages) {
            if (!msg || typeof msg !== 'object') continue;
            const msgObj = msg as Record<string, unknown>;
            const role = msgObj.role;
            
            // Apply capture filter
            if (cfg.autoCaptureFilter === 'user_only' && role !== 'user') continue;
            if (role !== 'user' && role !== 'assistant') continue;

            let text = '';
            const content = msgObj.content;
            if (typeof content === 'string') {
              text = content;
            } else if (Array.isArray(content)) {
              for (const block of content) {
                if (block && typeof block === 'object' && 'type' in block) {
                  const b = block as Record<string, unknown>;
                  if (b.type === 'text' && typeof b.text === 'string') {
                    text += (text ? '\n' : '') + b.text;
                  }
                }
              }
            }

            if (!text) continue;
            if (text.includes('<open-tasks>')) continue;

            conversationParts.push(`${role}: ${text.slice(0, 500)}`);
          }

          if (conversationParts.length < 2) return;

          const conversation = conversationParts.join('\n\n');
          const detection = await detector.detect(conversation);

          if (!detection.hasTasks || detection.confidence < cfg.detectionThreshold) return;

          // Auto-create new tasks
          if (cfg.autoDetect && detection.tasks.length > 0) {
            for (const task of detection.tasks) {
              try {
                const id = await client.create(task.title, {
                  type: task.type,
                  priority: task.priority || cfg.defaultPriority,
                  description: task.description,
                });
                api.logger.info(`openclaw-beads-unified: auto-created ${id}: "${task.title}"`);
                
                if (graphitiLinker) {
                  await graphitiLinker.linkTaskEvent('created', { 
                    id, 
                    title: task.title, 
                    type: task.type, 
                    priority: task.priority || cfg.defaultPriority, 
                    status: 'open' 
                  });
                }
              } catch (err) {
                api.logger.warn(`openclaw-beads-unified: failed to create task: ${String(err)}`);
              }
            }
          }

          // Auto-close completed tasks
          if (cfg.autoClose && detection.completedTasks.length > 0) {
            const openTasks = await client.list('open');
            const inProgressTasks = await client.list('in_progress');
            const allTasks = [...openTasks, ...inProgressTasks];

            for (const completed of detection.completedTasks) {
              const pattern = completed.pattern.toLowerCase();
              const matchingTask = allTasks.find(t => 
                t.title.toLowerCase().includes(pattern) ||
                t.id.toLowerCase().includes(pattern) ||
                pattern.includes(t.id.toLowerCase())
              );

              if (matchingTask) {
                try {
                  await client.close(matchingTask.id, completed.summary);
                  api.logger.info(`openclaw-beads-unified: auto-closed ${matchingTask.id}`);
                  
                  if (graphitiLinker) {
                    await graphitiLinker.linkTaskEvent('closed', { ...matchingTask, status: 'closed' }, completed.summary);
                  }
                } catch (err) {
                  api.logger.warn(`openclaw-beads-unified: failed to close task: ${String(err)}`);
                }
              }
            }
          }
        } catch (err) {
          api.logger.warn(`openclaw-beads-unified: detection failed: ${String(err)}`);
        }
      });
    }

    // ========================================================================
    // Service
    // ========================================================================

    api.registerService({
      id: 'openclaw-beads-unified',
      start: async () => {
        try {
          const tasks = await client.list('open');
          api.logger.info(`openclaw-beads-unified: connected to ${cfg.workspace} (${tasks.length} open tasks)`);
        } catch (err) {
          api.logger.warn(`openclaw-beads-unified: workspace check failed: ${String(err)}`);
        }
      },
      stop: () => {
        api.logger.info('openclaw-beads-unified: stopped');
      },
    });
  },
};

export default beadsUnifiedPlugin;
