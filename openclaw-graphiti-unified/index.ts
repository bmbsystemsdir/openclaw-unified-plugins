/**
 * OpenClaw Graphiti Plugin v3 (Unified) - Master Memory Architecture
 *
 * Merged implementation combining features from both Blitz and Dexter plugins.
 * 
 * Features:
 * - Temporal episodic memory with timestamps and recency weighting
 * - Explicit relationship types (decided, works_on, depends_on, contradicts, updates, etc.)
 * - Automatic entity extraction (people, projects, systems, concepts)
 * - Fact versioning with current vs superseded separation
 * - Dual contradiction detection: pattern-based (fast) + LLM (fallback)
 * - Goal tracking with [GOAL] tagging and status tracking
 * - User intent inference (goals, preferences, concerns)
 * - Temporal point-in-time queries
 * - Entity-centric relationship queries
 * - Configurable autoCapture filter (user_only option to prevent feedback loops)
 *
 * Tools:
 * - graphiti_search: Search episodes, entities, and facts
 * - graphiti_store: Manually store episodes
 * - graphiti_entities: List known entities
 * - graphiti_timeline: Get chronological episode history
 * - graphiti_temporal_query: Query facts at a specific point in time
 * - graphiti_relationships: Get entity relationships
 * - graphiti_contradictions: Check for contradictions with new content
 */

import { Type } from "@sinclair/typebox";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

// ============================================================================
// Types
// ============================================================================

type EpisodeType = "decision" | "event" | "learning" | "relationship_change" | "goal" | "preference" | "general";
type RelationshipType = "decided" | "works_on" | "depends_on" | "contradicts" | "updates" | "relates_to" | "wants" | "prefers" | "blocked_by" | "achieved" | "supersedes";
type IntentType = "decision" | "preference" | "goal" | "problem" | "completion" | "change";
type CaptureFilter = "all" | "user_only";

interface GraphitiConfig {
  endpoint: string;
  groupId: string;
  autoCapture: boolean;
  autoCaptureFilter: CaptureFilter;
  autoRecall: boolean;
  detectContradictions: boolean;
  captureThreshold: number;
  recallTopK: number;
  recencyBoostDays: number;
  llmProvider: string;
  llmModel: string;
  useLlmContradictionFallback: boolean;
}

interface Episode {
  uuid?: string;
  name: string;
  content: string;
  source: string;
  source_description: string;
  created_at?: string;
  valid_at?: string;
  entity_edges?: EntityEdge[];
}

interface Entity {
  uuid: string;
  name: string;
  entity_type: string;
  summary?: string;
  created_at?: string;
}

interface EntityEdge {
  uuid?: string;
  source_entity_uuid?: string;
  target_entity_uuid?: string;
  source_entity_name?: string;
  target_entity_name?: string;
  relation_type?: string;
  name?: string;
  fact?: string;
  created_at?: string;
  valid_at?: string;
  invalid_at?: string;
  expired_at?: string;
}

interface ClassificationResult {
  shouldCapture: boolean;
  episodeType: EpisodeType;
  intents: IntentType[];
  summary: string;
  entities: Array<{ name: string; type: string }>;
  relationships: Array<{ source: string; target: string; type: RelationshipType; fact: string }>;
  userIntent?: {
    goals: string[];
    preferences: string[];
    concerns: string[];
  };
  confidence: number;
}

interface ContradictionAlert {
  newContent: string;
  conflictingFact: EntityEdge;
  severity: 'low' | 'medium' | 'high';
}

interface GoalInsight {
  goal: string;
  status: 'active' | 'achieved' | 'blocked';
  relatedBeads: string[];
  blockers: string[];
}

// ============================================================================
// MCP Client (Extended)
// ============================================================================

class GraphitiMCPClient {
  private sessionId: string | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(
    private readonly endpoint: string,
    private readonly groupId: string,
    private readonly logger: { info: (msg: string) => void; warn: (msg: string) => void; error: (msg: string) => void },
  ) {}

  private async ensureSession(): Promise<void> {
    if (this.sessionId) return;
    
    // Prevent concurrent initialization
    if (this.initPromise) {
      await this.initPromise;
      return;
    }

    this.initPromise = (async () => {
      const url = this.endpoint.replace(/\/$/, '');
      try {
        const response = await fetch(`${url}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream'
          },
          body: JSON.stringify({
            jsonrpc: '2.0',
            id: Date.now(),
            method: 'initialize',
            params: {
              protocolVersion: '2024-11-05',
              capabilities: {},
              clientInfo: { name: 'openclaw-graphiti', version: '3.0.0' }
            }
          })
        });

        if (!response.ok) {
          throw new Error(`Session init failed: HTTP ${response.status}`);
        }

        const sessionId = response.headers.get('mcp-session-id');
        if (!sessionId) {
          throw new Error('No mcp-session-id returned from initialize');
        }

        this.sessionId = sessionId;
        this.logger.info(`Graphiti session established: ${sessionId.substring(0, 8)}...`);
      } catch (err) {
        this.initPromise = null; // Allow retry
        throw err;
      }
    })();

    await this.initPromise;
  }

  private parseSSE(text: string): unknown {
    // Parse Server-Sent Events format
    // Format: "event: message\ndata: <json>\n\n"
    const lines = text.trim().split('\n');
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const jsonStr = line.substring(6); // Remove "data: " prefix
        return JSON.parse(jsonStr);
      }
    }
    throw new Error('No data line found in SSE response');
  }

  private async call(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    await this.ensureSession();
    
    const url = this.endpoint.replace(/\/$/, '');
    
    // Prepare arguments with group_id
    const args = { group_id: this.groupId, ...params };
    
    try {
      const response = await fetch(`${url}`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json, text/event-stream',
          'mcp-session-id': this.sessionId!
        },
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: Date.now(),
          method: 'tools/call',
          params: {
            name: method,
            arguments: args
          },
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const contentType = response.headers.get('content-type') || '';
      let json: { result?: unknown; error?: { message: string } };

      if (contentType.includes('text/event-stream')) {
        // Parse SSE format
        const text = await response.text();
        json = this.parseSSE(text) as { result?: unknown; error?: { message: string } };
      } else {
        // Parse plain JSON
        json = await response.json() as { result?: unknown; error?: { message: string } };
      }

      if (json.error) throw new Error(json.error.message);
      
      // Extract structured content from MCP response
      const result = json.result as any;
      if (result?.structuredContent?.result) {
        return result.structuredContent.result;
      }
      return result;
    } catch (err) {
      this.logger.error(`Graphiti MCP call failed (${method}): ${String(err)}`);
      throw err;
    }
  }

  async addEpisode(episode: {
    name: string;
    content: string;
    sourceDescription?: string;
    relationships?: Array<{ source: string; target: string; type: string; fact: string }>;
  }): Promise<unknown> {
    let enrichedContent = episode.content;
    
    if (episode.relationships && episode.relationships.length > 0) {
      enrichedContent += '\n\nExplicit relationships:\n';
      for (const rel of episode.relationships) {
        enrichedContent += `- ${rel.source} [${rel.type}] ${rel.target}: ${rel.fact}\n`;
      }
    }
    
    return this.call('add_memory', {
      name: episode.name,
      episode_body: enrichedContent,
      source: 'text',
      source_description: episode.sourceDescription || 'agent conversation',
    });
  }

  async searchNodes(query: string, maxNodes = 5): Promise<Entity[]> {
    const result = await this.call('search_nodes', { 
      query, 
      group_ids: [this.groupId],
      max_nodes: maxNodes 
    }) as { nodes?: Entity[] } | null;
    return result?.nodes ?? [];
  }

  async searchFacts(query: string, maxFacts = 10, entityUuid?: string): Promise<EntityEdge[]> {
    const params: Record<string, unknown> = { 
      query, 
      group_ids: [this.groupId],
      max_facts: maxFacts 
    };
    if (entityUuid) params.center_node_uuid = entityUuid;
    const result = await this.call('search_memory_facts', params) as { facts?: EntityEdge[] } | null;
    return result?.facts ?? [];
  }

  async getEpisodes(maxEpisodes = 10): Promise<Episode[]> {
    // Note: base call() adds group_id, but get_episodes needs group_ids (array)
    // We'll pass group_ids and the server will use it instead
    const result = await this.call('get_episodes', { 
      group_ids: [this.groupId], 
      max_episodes: maxEpisodes 
    }) as { episodes?: Episode[] } | null;
    return result?.episodes ?? [];
  }

  async getStatus(): Promise<{ healthy: boolean; message?: string }> {
    try {
      await this.call('get_status', {});
      return { healthy: true };
    } catch (err) {
      return { healthy: false, message: String(err) };
    }
  }

  // === Extended Methods (from Dexter) ===

  async resolveEntityUuid(name: string): Promise<string | null> {
    const nodes = await this.searchNodes(name, 5);
    const exact = nodes.find(n => n.name.toLowerCase() === name.toLowerCase());
    return exact?.uuid || nodes[0]?.uuid || null;
  }

  async getEntityRelationships(entityName: string): Promise<{ entity: Entity | null; facts: EntityEdge[] }> {
    const nodes = await this.searchNodes(entityName, 1);
    const entity = nodes[0] || null;
    if (!entity) return { entity: null, facts: [] };
    const facts = await this.searchFacts(entityName, 20, entity.uuid);
    return { entity, facts };
  }

  async queryAtTime(query: string, date: Date): Promise<EntityEdge[]> {
    const allFacts = await this.searchFacts(query, 50);
    return allFacts.filter(fact => {
      const validAt = fact.valid_at ? new Date(fact.valid_at) : null;
      const invalidAt = fact.invalid_at ? new Date(fact.invalid_at) : null;
      if (!validAt) return false;
      if (validAt > date) return false;
      if (invalidAt && invalidAt <= date) return false;
      return true;
    });
  }

  async getFactHistory(topic: string): Promise<{ current: EntityEdge[]; superseded: EntityEdge[] }> {
    const allFacts = await this.searchFacts(topic, 50);
    const current = allFacts.filter(f => !f.invalid_at && !f.expired_at);
    const superseded = allFacts.filter(f => f.invalid_at || f.expired_at);
    return { current, superseded };
  }

  // === Goal Tracking (from Dexter) ===

  async extractAndStoreGoal(content: string, intents: IntentType[]): Promise<void> {
    if (!intents.includes('goal')) return;
    
    const goalPatterns = [
      /goal[:\s]+(.+?)(?:\.|$)/i,
      /need to (.+?)(?:\.|$)/i,
      /must (.+?)(?:\.|$)/i,
      /by ([A-Za-z]+ \d+)[,\s]+(.+?)(?:\.|$)/i,
      /deadline[:\s]+(.+?)(?:\.|$)/i,
    ];
    
    for (const pattern of goalPatterns) {
      const match = content.match(pattern);
      if (match) {
        const goalText = match[1] || match[2] || match[0];
        await this.addEpisode({
          name: `goal-${Date.now()}`,
          content: `[GOAL] ${goalText}`,
          sourceDescription: 'auto-extracted goal',
        });
        break;
      }
    }
  }

  async getGoalInsights(): Promise<GoalInsight[]> {
    const facts = await this.searchFacts('goal deadline objective must need to', 30);
    const goalFacts = facts.filter(f => 
      f.fact?.toLowerCase().includes('goal') ||
      f.fact?.toLowerCase().includes('deadline') ||
      f.fact?.toLowerCase().includes('must') ||
      f.fact?.includes('[GOAL]')
    );
    
    const insights: GoalInsight[] = [];
    for (const fact of goalFacts) {
      const isSuperseded = fact.invalid_at || fact.expired_at;
      insights.push({
        goal: (fact.fact || '').replace('[GOAL]', '').trim(),
        status: isSuperseded ? 'achieved' : 'active',
        relatedBeads: [],
        blockers: [],
      });
    }
    return insights;
  }
}

// ============================================================================
// Contradiction Detection (Dual: Pattern + LLM Fallback)
// ============================================================================

class ContradictionDetector {
  constructor(
    private readonly client: GraphitiMCPClient,
    private readonly llmModel: string,
    private readonly useLlmFallback: boolean,
    private readonly logger: { info: (msg: string) => void; warn: (msg: string) => void },
  ) {}

  /**
   * Pattern-based contradiction detection (fast, no API calls)
   */
  async checkPatternBased(newContent: string): Promise<ContradictionAlert[]> {
    const alerts: ContradictionAlert[] = [];
    const relatedFacts = await this.client.searchFacts(newContent, 20);
    const currentFacts = relatedFacts.filter(f => !f.invalid_at && !f.expired_at);
    const newLower = newContent.toLowerCase();
    
    for (const fact of currentFacts) {
      const factLower = (fact.fact || '').toLowerCase();
      
      // Check for negation patterns
      const hasNegation = (
        (newLower.includes('not') && !factLower.includes('not')) ||
        (!newLower.includes('not') && factLower.includes('not')) ||
        (newLower.includes("don't") && !factLower.includes("don't")) ||
        (newLower.includes('instead') || newLower.includes('rather than'))
      );
      
      // Check for change patterns
      const hasChange = (
        newLower.includes('changed') ||
        newLower.includes('updated') ||
        newLower.includes('switched') ||
        newLower.includes('replaced') ||
        newLower.includes('no longer')
      );
      
      // Check for decision reversals
      const isDecisionReversal = (
        fact.name === 'DECIDED' && 
        (newLower.includes('decided') || newLower.includes('chose'))
      );
      
      if (hasNegation || hasChange || isDecisionReversal) {
        const severity = isDecisionReversal ? 'high' : hasChange ? 'medium' : 'low';
        alerts.push({
          newContent: newContent.slice(0, 100),
          conflictingFact: fact,
          severity,
        });
      }
    }
    
    return alerts;
  }

  /**
   * LLM-based contradiction detection (more accurate, uses API)
   */
  async checkLlmBased(newContent: string, existingFacts: EntityEdge[]): Promise<ContradictionAlert[]> {
    if (existingFacts.length === 0) return [];

    const systemPrompt = `You are a contradiction detector. Compare new information against existing facts and identify conflicts.

Respond with JSON only:
{
  "hasContradiction": boolean,
  "conflicts": [
    { "newFact": "the new information", "existingFact": "the existing fact it conflicts with", "severity": "low" | "medium" | "high" }
  ]
}`;

    try {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) return [];

      const existingFactsText = existingFacts
        .filter(f => !f.invalid_at)
        .map(f => `- ${f.fact || f.name}`)
        .join('\n');

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
            { role: 'user', content: `New information:\n${newContent}\n\nExisting facts:\n${existingFactsText}` },
          ],
          temperature: 0.1,
          max_tokens: 500,
        }),
      });

      if (!response.ok) return [];

      const json = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
      const content = json.choices?.[0]?.message?.content;
      if (!content) return [];

      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (!jsonMatch) return [];

      const result = JSON.parse(jsonMatch[0]) as {
        hasContradiction: boolean;
        conflicts: Array<{ newFact: string; existingFact: string; severity: 'low' | 'medium' | 'high' }>;
      };

      if (!result.hasContradiction) return [];

      return result.conflicts.map(c => ({
        newContent: c.newFact,
        conflictingFact: { fact: c.existingFact } as EntityEdge,
        severity: c.severity,
      }));
    } catch (err) {
      this.logger.warn(`LLM contradiction check failed: ${String(err)}`);
      return [];
    }
  }

  /**
   * Combined detection: pattern-based first, LLM fallback if enabled and no patterns found
   */
  async check(newContent: string): Promise<ContradictionAlert[]> {
    // Try pattern-based first (fast)
    const patternAlerts = await this.checkPatternBased(newContent);
    
    if (patternAlerts.length > 0) {
      this.logger.info(`Contradiction detected (pattern): ${patternAlerts.length} conflicts`);
      return patternAlerts;
    }

    // If no pattern matches and LLM fallback is enabled, try LLM
    if (this.useLlmFallback) {
      const existingFacts = await this.client.searchFacts(newContent, 10);
      const llmAlerts = await this.checkLlmBased(newContent, existingFacts);
      if (llmAlerts.length > 0) {
        this.logger.info(`Contradiction detected (LLM): ${llmAlerts.length} conflicts`);
      }
      return llmAlerts;
    }

    return [];
  }
}

// ============================================================================
// Intelligent Classifier
// ============================================================================

class IntelligentClassifier {
  constructor(
    private readonly model: string,
    private readonly logger: { info: (msg: string) => void; warn: (msg: string) => void },
  ) {}

  async classify(conversation: string): Promise<ClassificationResult> {
    const systemPrompt = `You are an intelligent memory system. Analyze conversations to:
1. IDENTIFY what should be remembered (decisions, events, learnings, goals, preferences)
2. EXTRACT entities with types (person, project, system, concept, organization)
3. DEFINE explicit relationships with types
4. INFER user intent (goals, preferences, concerns)
5. DETECT intents: decision, preference, goal, problem, completion, change

Episode types: decision, event, learning, relationship_change, goal, preference, general
Relationship types: decided, works_on, depends_on, contradicts, updates, relates_to, wants, prefers, blocked_by, achieved, supersedes

Respond with JSON only:
{
  "shouldCapture": boolean,
  "episodeType": "decision" | "event" | "learning" | "relationship_change" | "goal" | "preference" | "general",
  "intents": ["decision", "preference", "goal", "problem", "completion", "change"],
  "summary": "Clear 1-2 sentence summary",
  "entities": [{"name": "entity name", "type": "person|project|system|concept|organization|process"}],
  "relationships": [{"source": "entity1", "target": "entity2", "type": "relationship_type", "fact": "statement"}],
  "userIntent": {"goals": [], "preferences": [], "concerns": []},
  "confidence": 0.0-1.0
}`;

    try {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) return this.fallbackClassify(conversation);

      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: this.model,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: `Analyze:\n\n${conversation}` },
          ],
          temperature: 0.2,
          max_tokens: 800,
        }),
      });

      if (!response.ok) throw new Error(`API error: ${response.status}`);

      const json = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
      const content = json.choices?.[0]?.message?.content;
      if (!content) throw new Error('No content');

      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error('No JSON');

      return JSON.parse(jsonMatch[0]) as ClassificationResult;
    } catch (err) {
      this.logger.warn(`Classification failed: ${String(err)}, using fallback`);
      return this.fallbackClassify(conversation);
    }
  }

  private fallbackClassify(conversation: string): ClassificationResult {
    const lower = conversation.toLowerCase();
    const intents: IntentType[] = [];
    
    if (/decided|decision|agreed|chose|selected/.test(lower)) intents.push('decision');
    if (/prefer|like|rather|favorite/.test(lower)) intents.push('preference');
    if (/goal|want to|need to|objective|aim/.test(lower)) intents.push('goal');
    if (/problem|issue|bug|error|failed/.test(lower)) intents.push('problem');
    if (/done|finished|completed|fixed|resolved/.test(lower)) intents.push('completion');
    if (/changed|updated|switched|replaced/.test(lower)) intents.push('change');
    
    const shouldCapture = intents.length > 0;
    let episodeType: EpisodeType = 'general';
    if (intents.includes('decision')) episodeType = 'decision';
    else if (intents.includes('goal')) episodeType = 'goal';
    else if (intents.includes('preference')) episodeType = 'preference';
    else if (intents.includes('completion')) episodeType = 'event';

    return {
      shouldCapture,
      episodeType,
      intents,
      summary: conversation.slice(0, 300) + (conversation.length > 300 ? '...' : ''),
      entities: [],
      relationships: [],
      userIntent: { goals: [], preferences: [], concerns: [] },
      confidence: shouldCapture ? 0.5 : 0.2,
    };
  }
}

// ============================================================================
// Config Parser
// ============================================================================

const graphitiConfigSchema = {
  parse(value: unknown): GraphitiConfig {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return {
        endpoint: 'http://localhost:8000/mcp/',
        groupId: 'default',
        autoCapture: true,
        autoCaptureFilter: 'all',
        autoRecall: true,
        detectContradictions: true,
        captureThreshold: 0.5,
        recallTopK: 8,
        recencyBoostDays: 7,
        llmProvider: 'openai',
        llmModel: 'gpt-4.1-nano',
        useLlmContradictionFallback: true,
      };
    }

    const cfg = value as Record<string, unknown>;

    return {
      endpoint: typeof cfg.endpoint === 'string' ? cfg.endpoint : 'http://localhost:8000/mcp/',
      groupId: typeof cfg.groupId === 'string' ? cfg.groupId : 'default',
      autoCapture: cfg.autoCapture !== false,
      autoCaptureFilter: cfg.autoCaptureFilter === 'user_only' ? 'user_only' : 'all',
      autoRecall: cfg.autoRecall !== false,
      detectContradictions: cfg.detectContradictions !== false,
      captureThreshold: typeof cfg.captureThreshold === 'number' ? cfg.captureThreshold : 0.5,
      recallTopK: typeof cfg.recallTopK === 'number' ? cfg.recallTopK : 8,
      recencyBoostDays: typeof cfg.recencyBoostDays === 'number' ? cfg.recencyBoostDays : 7,
      llmProvider: typeof cfg.llmProvider === 'string' ? cfg.llmProvider : 'openai',
      llmModel: typeof cfg.llmModel === 'string' ? cfg.llmModel : 'gpt-4.1-nano',
      useLlmContradictionFallback: cfg.useLlmContradictionFallback !== false,
    };
  },
};

// ============================================================================
// Helpers
// ============================================================================

function formatTimestamp(date?: string): string {
  if (!date) return 'unknown';
  try {
    return new Date(date).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
  } catch {
    return date;
  }
}

function getDaysAgo(date?: string): number {
  if (!date) return 999;
  try {
    return Math.floor((Date.now() - new Date(date).getTime()) / (1000 * 60 * 60 * 24));
  } catch {
    return 999;
  }
}

function formatFactWithRecency(fact: EntityEdge, recencyBoostDays: number): string {
  const daysAgo = getDaysAgo(fact.created_at);
  const recencyMarker = daysAgo <= recencyBoostDays ? ' 🕐' : '';
  const validStatus = fact.invalid_at ? ' [OUTDATED]' : '';
  const dateStr = fact.created_at ? ` (${formatTimestamp(fact.created_at)})` : '';
  return `• ${fact.fact || fact.name}${dateStr}${recencyMarker}${validStatus}`;
}

// ============================================================================
// Plugin Definition
// ============================================================================

const graphitiUnifiedPlugin = {
  id: 'openclaw-graphiti-unified',
  name: 'Graphiti Knowledge Graph (Unified)',
  description: 'Unified temporal knowledge graph with pattern+LLM contradiction detection, goal tracking, and temporal queries',
  kind: 'extension' as const,
  configSchema: graphitiConfigSchema,

  register(api: OpenClawPluginApi) {
    const cfg = graphitiConfigSchema.parse(api.pluginConfig);
    const client = new GraphitiMCPClient(cfg.endpoint, cfg.groupId, api.logger);
    const classifier = new IntelligentClassifier(cfg.llmModel, api.logger);
    const contradictionDetector = new ContradictionDetector(client, cfg.llmModel, cfg.useLlmContradictionFallback, api.logger);

    api.logger.info(
      `openclaw-graphiti-unified: registered (endpoint: ${cfg.endpoint}, group: ${cfg.groupId}, captureFilter: ${cfg.autoCaptureFilter}, contradictions: pattern+${cfg.useLlmContradictionFallback ? 'llm' : 'none'})`,
    );

    // ========================================================================
    // Tools (existing + new)
    // ========================================================================

    api.registerTool(
      {
        name: 'graphiti_search',
        label: 'Graphiti Search',
        description: 'Search the knowledge graph for episodes, entities, and facts.',
        parameters: Type.Object({
          query: Type.String({ description: 'Search query' }),
          maxResults: Type.Optional(Type.Number({ description: 'Maximum results per category (default: 5)' })),
          includeEntities: Type.Optional(Type.Boolean({ description: 'Include entity nodes' })),
          includeFacts: Type.Optional(Type.Boolean({ description: 'Include relationship facts' })),
        }),
        async execute(_toolCallId, params) {
          const { query, maxResults = 5, includeEntities = true, includeFacts = true } = params as {
            query: string; maxResults?: number; includeEntities?: boolean; includeFacts?: boolean;
          };

          try {
            const results: string[] = [];

            if (includeEntities) {
              const nodes = await client.searchNodes(query, maxResults);
              if (nodes.length > 0) {
                results.push('**Entities:**');
                for (const node of nodes) {
                  results.push(`• ${node.name} (${node.entity_type}): ${node.summary || 'No summary'}`);
                }
              }
            }

            if (includeFacts) {
              const facts = await client.searchFacts(query, maxResults);
              if (facts.length > 0) {
                results.push('\n**Facts/Relationships:**');
                for (const fact of facts) {
                  results.push(formatFactWithRecency(fact, cfg.recencyBoostDays));
                }
              }
            }

            if (results.length === 0) {
              return { content: [{ type: 'text', text: 'No results found.' }], details: { count: 0 } };
            }

            return { content: [{ type: 'text', text: results.join('\n') }], details: { query } };
          } catch (err) {
            return { content: [{ type: 'text', text: `Search failed: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'graphiti_search' },
    );

    api.registerTool(
      {
        name: 'graphiti_store',
        label: 'Graphiti Store',
        description: 'Manually store an episode in the knowledge graph.',
        parameters: Type.Object({
          content: Type.String({ description: 'Episode content' }),
          type: Type.Optional(Type.Union([
            Type.Literal('decision'), Type.Literal('event'), Type.Literal('learning'), Type.Literal('general'),
          ], { description: 'Episode type' })),
          title: Type.Optional(Type.String({ description: 'Episode title' })),
        }),
        async execute(_toolCallId, params) {
          const { content, type = 'general', title } = params as { content: string; type?: EpisodeType; title?: string };

          try {
            const episodeTitle = title || `${type.charAt(0).toUpperCase() + type.slice(1)}: ${content.slice(0, 50)}...`;
            await client.addEpisode({ name: episodeTitle, content, sourceDescription: `manual ${type}` });
            return { content: [{ type: 'text', text: `Stored ${type} episode: "${episodeTitle}"` }], details: { type, title: episodeTitle } };
          } catch (err) {
            return { content: [{ type: 'text', text: `Store failed: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'graphiti_store' },
    );

    api.registerTool(
      {
        name: 'graphiti_entities',
        label: 'Graphiti Entities',
        description: 'List known entities in the knowledge graph.',
        parameters: Type.Object({
          filter: Type.Optional(Type.String({ description: 'Filter by name or type' })),
          limit: Type.Optional(Type.Number({ description: 'Max entities (default: 20)' })),
        }),
        async execute(_toolCallId, params) {
          const { filter, limit = 20 } = params as { filter?: string; limit?: number };

          try {
            const nodes = await client.searchNodes(filter || '*', limit);
            if (nodes.length === 0) {
              return { content: [{ type: 'text', text: 'No entities found.' }], details: { count: 0 } };
            }

            const byType: Record<string, Entity[]> = {};
            for (const node of nodes) {
              const t = node.entity_type || 'unknown';
              if (!byType[t]) byType[t] = [];
              byType[t].push(node);
            }

            const lines: string[] = [];
            for (const [t, entities] of Object.entries(byType)) {
              lines.push(`**${t}:**`);
              for (const e of entities) {
                lines.push(`  • ${e.name}${e.summary ? `: ${e.summary}` : ''}`);
              }
            }

            return { content: [{ type: 'text', text: lines.join('\n') }], details: { count: nodes.length } };
          } catch (err) {
            return { content: [{ type: 'text', text: `Entities failed: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'graphiti_entities' },
    );

    api.registerTool(
      {
        name: 'graphiti_timeline',
        label: 'Graphiti Timeline',
        description: 'Get chronological history of episodes.',
        parameters: Type.Object({
          topic: Type.Optional(Type.String({ description: 'Filter by topic' })),
          limit: Type.Optional(Type.Number({ description: 'Max episodes (default: 10)' })),
        }),
        async execute(_toolCallId, params) {
          const { topic, limit = 10 } = params as { topic?: string; limit?: number };

          try {
            let episodes = await client.getEpisodes(limit);
            if (topic) {
              const lower = topic.toLowerCase();
              episodes = episodes.filter(e => e.name?.toLowerCase().includes(lower) || e.content?.toLowerCase().includes(lower));
            }

            if (episodes.length === 0) {
              return { content: [{ type: 'text', text: topic ? `No episodes matching "${topic}".` : 'No episodes found.' }], details: { count: 0 } };
            }

            episodes.sort((a, b) => {
              const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
              const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
              return dateB - dateA;
            });

            const lines = episodes.map(e => `[${formatTimestamp(e.created_at)}] ${e.name}: ${e.content?.slice(0, 200) || ''}`);
            return { content: [{ type: 'text', text: `**Timeline (${episodes.length}):**\n\n${lines.join('\n\n')}` }], details: { count: episodes.length } };
          } catch (err) {
            return { content: [{ type: 'text', text: `Timeline failed: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'graphiti_timeline' },
    );

    // === New Tools (from Dexter) ===

    api.registerTool(
      {
        name: 'graphiti_temporal_query',
        label: 'Graphiti Temporal Query',
        description: 'Query facts as they were at a specific point in time.',
        parameters: Type.Object({
          query: Type.String({ description: 'Search query' }),
          date: Type.String({ description: 'ISO date string (e.g., 2026-01-15)' }),
        }),
        async execute(_toolCallId, params) {
          const { query, date } = params as { query: string; date: string };

          try {
            const targetDate = new Date(date);
            const facts = await client.queryAtTime(query, targetDate);
            
            if (facts.length === 0) {
              return { content: [{ type: 'text', text: `No facts found for "${query}" as of ${date}.` }], details: { count: 0 } };
            }

            const lines = facts.map(f => formatFactWithRecency(f, cfg.recencyBoostDays));
            return { content: [{ type: 'text', text: `**Facts as of ${date}:**\n\n${lines.join('\n')}` }], details: { count: facts.length } };
          } catch (err) {
            return { content: [{ type: 'text', text: `Temporal query failed: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'graphiti_temporal_query' },
    );

    api.registerTool(
      {
        name: 'graphiti_relationships',
        label: 'Graphiti Relationships',
        description: 'Get all relationships for a specific entity.',
        parameters: Type.Object({
          entity: Type.String({ description: 'Entity name' }),
        }),
        async execute(_toolCallId, params) {
          const { entity } = params as { entity: string };

          try {
            const { entity: foundEntity, facts } = await client.getEntityRelationships(entity);
            
            if (!foundEntity) {
              return { content: [{ type: 'text', text: `Entity "${entity}" not found.` }], details: { found: false } };
            }

            const lines = [`**${foundEntity.name}** (${foundEntity.entity_type})`, foundEntity.summary || ''];
            if (facts.length > 0) {
              lines.push('\n**Relationships:**');
              for (const fact of facts) {
                lines.push(formatFactWithRecency(fact, cfg.recencyBoostDays));
              }
            }

            return { content: [{ type: 'text', text: lines.join('\n') }], details: { entity: foundEntity.name, factCount: facts.length } };
          } catch (err) {
            return { content: [{ type: 'text', text: `Relationships failed: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'graphiti_relationships' },
    );

    api.registerTool(
      {
        name: 'graphiti_contradictions',
        label: 'Check Contradictions',
        description: 'Check if new content contradicts existing facts.',
        parameters: Type.Object({
          content: Type.String({ description: 'New content to check' }),
        }),
        async execute(_toolCallId, params) {
          const { content } = params as { content: string };

          try {
            const alerts = await contradictionDetector.check(content);
            
            if (alerts.length === 0) {
              return { content: [{ type: 'text', text: 'No contradictions detected.' }], details: { count: 0 } };
            }

            const lines = ['**Potential contradictions:**'];
            for (const alert of alerts) {
              lines.push(`• [${alert.severity.toUpperCase()}] New: "${alert.newContent}" conflicts with: "${alert.conflictingFact.fact}"`);
            }

            return { content: [{ type: 'text', text: lines.join('\n') }], details: { count: alerts.length } };
          } catch (err) {
            return { content: [{ type: 'text', text: `Contradiction check failed: ${String(err)}` }], details: { error: String(err) } };
          }
        },
      },
      { name: 'graphiti_contradictions' },
    );

    // ========================================================================
    // Lifecycle Hooks
    // ========================================================================

    // Auto-recall with recency weighting
    if (cfg.autoRecall) {
      api.on('before_agent_start', async (event) => {
        if (!event.prompt || event.prompt.length < 10) return;

        try {
          const facts = await client.searchFacts(event.prompt, cfg.recallTopK);
          const nodes = await client.searchNodes(event.prompt, Math.ceil(cfg.recallTopK / 2));

          if (facts.length === 0 && nodes.length === 0) return;

          const contextParts: string[] = [];

          if (facts.length > 0) {
            const sortedFacts = [...facts].sort((a, b) => getDaysAgo(a.created_at) - getDaysAgo(b.created_at));
            const recentFacts = sortedFacts.filter(f => getDaysAgo(f.created_at) <= cfg.recencyBoostDays && !f.invalid_at);
            const olderFacts = sortedFacts.filter(f => getDaysAgo(f.created_at) > cfg.recencyBoostDays && !f.invalid_at);

            if (recentFacts.length > 0) {
              contextParts.push(`**Recent knowledge** (last ${cfg.recencyBoostDays} days):`);
              for (const fact of recentFacts) contextParts.push(formatFactWithRecency(fact, cfg.recencyBoostDays));
            }

            if (olderFacts.length > 0) {
              contextParts.push('\n**Background knowledge:**');
              for (const fact of olderFacts.slice(0, 5)) contextParts.push(formatFactWithRecency(fact, cfg.recencyBoostDays));
            }
          }

          if (nodes.length > 0) {
            contextParts.push('\n**Known entities:**');
            for (const node of nodes) {
              contextParts.push(`• ${node.name} (${node.entity_type})${node.summary ? `: ${node.summary}` : ''}`);
            }
          }

          api.logger.info(`openclaw-graphiti-unified: injecting ${facts.length} facts, ${nodes.length} entities`);

          return { systemContext: `<knowledge-graph>\n${contextParts.join('\n')}\n</knowledge-graph>` };
        } catch (err) {
          api.logger.warn(`openclaw-graphiti-unified: recall failed: ${String(err)}`);
        }
      });
    }

    // Auto-capture with filter option
    if (cfg.autoCapture) {
      api.on('agent_end', async (event) => {
        if (!event.success || !event.messages || event.messages.length === 0) return;

        try {
          const recentMessages = event.messages.slice(-8);
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
            if (text.includes('<knowledge-graph>') || text.includes('<relevant-memories>') || text.includes('<open-tasks>')) continue;

            conversationParts.push(`${role}: ${text.slice(0, 600)}`);
          }

          if (conversationParts.length < 2) return;

          const conversation = conversationParts.join('\n\n');
          const classification = await classifier.classify(conversation);

          if (!classification.shouldCapture || classification.confidence < cfg.captureThreshold) {
            api.logger.info(`openclaw-graphiti-unified: skipping (confidence: ${classification.confidence.toFixed(2)})`);
            return;
          }

          // Check for contradictions
          if (cfg.detectContradictions && classification.summary) {
            const alerts = await contradictionDetector.check(classification.summary);
            if (alerts.length > 0) {
              api.logger.info(`openclaw-graphiti-unified: ${alerts.length} contradiction(s) detected`);
              for (const alert of alerts) {
                classification.relationships.push({
                  source: 'new information',
                  target: 'previous fact',
                  type: alert.severity === 'high' ? 'contradicts' : 'updates',
                  fact: `${alert.newContent} (${alert.severity} change)`,
                });
              }
            }
          }

          // Extract goals if detected
          if (classification.intents.includes('goal')) {
            await client.extractAndStoreGoal(conversation, classification.intents);
          }

          // Store episode
          let episodeContent = classification.summary;
          if (classification.userIntent) {
            const intent = classification.userIntent;
            if (intent.goals.length > 0) episodeContent += `\n\nUser goals: ${intent.goals.join('; ')}`;
            if (intent.preferences.length > 0) episodeContent += `\nUser preferences: ${intent.preferences.join('; ')}`;
            if (intent.concerns.length > 0) episodeContent += `\nUser concerns: ${intent.concerns.join('; ')}`;
          }

          await client.addEpisode({
            name: `${classification.episodeType}: ${classification.summary.slice(0, 80)}`,
            content: episodeContent,
            sourceDescription: `auto-captured ${classification.episodeType}`,
            relationships: classification.relationships,
          });

          api.logger.info(
            `openclaw-graphiti-unified: captured ${classification.episodeType} (confidence: ${classification.confidence.toFixed(2)}, intents: ${classification.intents.join(',')})`,
          );
        } catch (err) {
          api.logger.warn(`openclaw-graphiti-unified: capture failed: ${String(err)}`);
        }
      });
    }

    // ========================================================================
    // Service
    // ========================================================================

    api.registerService({
      id: 'openclaw-graphiti-unified',
      start: async () => {
        const status = await client.getStatus();
        if (status.healthy) {
          api.logger.info(`openclaw-graphiti-unified: connected to ${cfg.endpoint}`);
        } else {
          api.logger.warn(`openclaw-graphiti-unified: connection failed: ${status.message}`);
        }
      },
      stop: () => {
        api.logger.info('openclaw-graphiti-unified: stopped');
      },
    });
  },
};

export default graphitiUnifiedPlugin;
