import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import dotenv from 'dotenv';
dotenv.config();
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { StateGraph, MessagesAnnotation } from '@langchain/langgraph';
import { ToolMessage } from '@langchain/core/messages';


const llm = new ChatGoogleGenerativeAI({
  model: 'gemini-1.5-flash',
});


const resumeSchema = z.object({
  resume: z.string().min(100).describe('The resume content to analyze'),
  focus: z.string().optional().describe('Specific area to focus on'),
});

// Appreciation Tool
const resumeAppreciation = tool(
  async ({ resume, focus }) => {
    const prompt = `**Appreciation Section**\nAppreciate this resume, highlighting key strengths and positive aspects.
    
    Resume Content:
    ${focus ? `Focus Area: ${focus}\n` : ''} 
    ${resume.substring(0, 2000)}... [truncated if too long]

    Provide:
    1. 3 key strengths or positive aspects
    2. Focus on experience and projects
    3. Use Hinglish
    word limit : 100 words in 3 points give and give heading appericiation at top
    `;

    const result = await llm.invoke(prompt);
    return result.content;
  },
  {
    name: 'resume_appreciation',
    description:
      'Appreciate the key strengths and positive aspects in Hinglish ',
    schema: resumeSchema,
  }
);

// Roast Tool
const resumeRoast = tool(
  async ({ resume, focus }) => {
    const prompt = `**Roast Section**\nRoast this resume, providing constructive criticism with 3 key areas for improvement in Hinglish:
    
    Resume Content:
    ${focus ? `Focus Area: ${focus}\n` : ''} 
    ${resume.substring(0, 2000)}... [truncated if too long]

    Provide:
    1. 3 key areas for improvement
    2. Specific examples of how to improve
    3. Use Hinglish for a light-hearted tone
     word limit : 100 words and give heading roast at top
    `;

    const result = await llm.invoke(prompt);
    return result.content;
  },
  {
    name: 'resume_roast',
    description: 'Provide constructive criticism in Hinglish',
    schema: resumeSchema,
  }
);

// Feedback Tool
const resumeFeedback = tool(
  async ({ resume, focus }) => {
    const prompt = `**Feedback Section**\nProvide professional feedback on this resume with suggestions for improvement in Hinglish:
    
    Resume Content:
    ${focus ? `Focus Area: ${focus}\n` : ''} 
    ${resume.substring(0, 2000)}... [truncated if too long]

    Provide:
    1. 3 actionable suggestions for improvement
    2. Focus on areas such as formatting, wording, and presentation
    3. Use Hinglish
     word limit : 100 words and give heading feedback at top
    `;

    const result = await llm.invoke(prompt);
    return result.content;
  },
  {
    name: 'resume_feedback',
    description: 'Provide actionable suggestions for improvement in Hinglish',
    schema: resumeSchema,
  }
);

// Tools setup
const tools = [resumeAppreciation, resumeRoast, resumeFeedback];
const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));
const llmWithTools = llm.bindTools(tools);

function detectExecutionOrder(message) {
  const keywordPatterns = [
    { pattern: /\b(roast)\b/i, name: 'roast' }, // Case-insensitive matching with word boundaries
    { pattern: /\b(appreciation)\b/i, name: 'appreciation' },
    { pattern: /\b(feedback)\b/i, name: 'feedback' },
  ];

  const executionOrder = [];
  const usedKeywords = new Set(); // Track used keywords to avoid duplicates

  // Find all matches in the message
  for (const { pattern, name } of keywordPatterns) {
    const matches = message.match(pattern);
    if (matches && !usedKeywords.has(name)) {
      executionOrder.push(name);
      usedKeywords.add(name);
    }
  }

  // If no specific request, default to feedback
  return executionOrder.length > 0 ? executionOrder : ['feedback'];
}

// LLM Node: Decides which tool(s) to use based on user input
async function llmCall(state) {
  const lastMessage = state.messages.at(-1).content.toLowerCase();
  const resume = `[${`Your resume content here`}]`;
  const focus = '';

  const resultMessages = [];

  // Detect execution order from user message
  const executionOrder = detectExecutionOrder(lastMessage);

  for (const action of executionOrder) {
    switch (action) {
      case 'appreciation':
        resultMessages.push(
          await toolsByName['resume_appreciation'].invoke({ resume, focus })
        );
        break;
      case 'roast':
        resultMessages.push(
          await toolsByName['resume_roast'].invoke({ resume, focus })
        );
        break;
      case 'feedback':
        resultMessages.push(
          await toolsByName['resume_feedback'].invoke({ resume, focus })
        );
        break;
    }
  }

  return { messages: resultMessages };
}

// Tool Node: Executes tool calls
async function toolNode(state) {
  const lastMessage = state.messages.at(-1);

  // If there are tool calls to execute, invoke them
  if (lastMessage?.tool_calls?.length) {
    const results = [];
    for (const toolCall of lastMessage.tool_calls) {
      const tool = toolsByName[toolCall.name];
      const observation = await tool.invoke(toolCall.args);
      results.push(
        new ToolMessage({
          content: observation,
          tool_call_id: toolCall.id,
        })
      );
    }
    return { messages: results };
  }

  return { messages: [] };
}

// Conditional logic for tool continuation
function shouldContinue(state) {
  const lastMessage = state.messages.at(-1);
  if (lastMessage?.tool_calls?.length) {
    return 'Action';
  }
  return '__end__';
}

// State Graph setup: Build the flow
const agentBuilder = new StateGraph(MessagesAnnotation)
  .addNode('llmCall', llmCall)
  .addNode('tools', toolNode)
  .addEdge('__start__', 'llmCall')
  .addConditionalEdges('llmCall', shouldContinue, {
    Action: 'tools',
    __end__: '__end__',
  })
  .addEdge('tools', 'llmCall')
  .compile({ recursionLimit: 3 });

// Example usage
const messages = [
  {
    role: 'user',
    content: `Please first give roast than appreciation than feedback on this resume :`,
  },
];

try {
  const result = await agentBuilder.invoke({ messages });
  const formattedOutput = result.messages
    .map((msg) => msg.content)
    .join('\n\n');
  console.log('Final Analysis:\n', formattedOutput);
} catch (error) {
  console.error('Analysis Error:', error);
}
