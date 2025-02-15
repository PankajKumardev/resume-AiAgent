import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import dotenv from 'dotenv';
dotenv.config();
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { StateGraph, MessagesAnnotation } from '@langchain/langgraph';
import { ToolMessage } from '@langchain/core/messages';

// Load LLM (Google Gemini 1.5)
const llm = new ChatGoogleGenerativeAI({
  model: 'gemini-1.5-flash',
});

// Define schema for resume analysis
const resumeSchema = z.object({
  resume: z.string().min(100).describe('The resume content to analyze'),
  focus: z.string().optional().describe('Specific area to focus on'),
});

// Appreciation Tool
const resumeAppreciation = tool(
  async ({ resume, focus }) => {
    const prompt = `Appreciate this resume, highlighting key strengths and positive aspects.
    
    Resume Content:
    ${focus ? `Focus Area: ${focus}\n` : ''} 
    ${resume.substring(0, 2000)}... [truncated if too long]

    Provide:
    1. 3 key strengths or positive aspects
    2. Focus on experience and projects
    3. Use Hinglish`;

    const result = await llm.invoke(prompt);
    return result.content;
  },
  {
    name: 'resume_appreciation',
    description:
      'Appreciate the key strengths and positive aspects in Hinglish',
    schema: resumeSchema,
  }
);

// Roast Tool
const resumeRoast = tool(
  async ({ resume, focus }) => {
    const prompt = `Roast this resume, providing constructive criticism with 3 key areas for improvement in Hinglish:
    
    Resume Content:
    ${focus ? `Focus Area: ${focus}\n` : ''} 
    ${resume.substring(0, 2000)}... [truncated if too long]

    Provide:
    1. 3 key areas for improvement
    2. Specific examples of how to improve
    3. Use Hinglish for a light-hearted tone`;

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
    const prompt = `Provide professional feedback on this resume with suggestions for improvement in Hinglish:
    
    Resume Content:
    ${focus ? `Focus Area: ${focus}\n` : ''} 
    ${resume.substring(0, 2000)}... [truncated if too long]

    Provide:
    1. 3 actionable suggestions for improvement
    2. Focus on areas such as formatting, wording, and presentation
    3. Use Hinglish`;

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

// LLM Node: Decides which tool(s) to use based on user input
async function llmCall(state) {
  const lastMessage = state.messages.at(-1).content.toLowerCase();
  const resume = ["your resume content here"];
  const focus = '';

  const resultMessages = [];

  // Check if the user wants appreciation, roast, or feedback
  if (lastMessage.includes('appreciation')) {
    resultMessages.push(
      await toolsByName['resume_appreciation'].invoke({ resume, focus })
    );
  }
  if (lastMessage.includes('roast')) {
    resultMessages.push(
      await toolsByName['resume_roast'].invoke({ resume, focus })
    );
  }
  if (lastMessage.includes('feedback')) {
    resultMessages.push(
      await toolsByName['resume_feedback'].invoke({ resume, focus })
    );
  }

  // If no specific request, respond with default feedback
  if (!resultMessages.length) {
    resultMessages.push({
      content: 'Please specify either appreciation, roast, or feedback.',
    });
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
    content: `Please give roast than appreciation this resume than feedback : `,
  },
];

try {
  const result = await agentBuilder.invoke({ messages });
  console.log('Final Analysis:', result.messages.at(-1).content);
} catch (error) {
  console.error('Analysis Error:', error);
}
