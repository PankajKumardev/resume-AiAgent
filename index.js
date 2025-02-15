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
  const resume = `[${`Pankaj Kumar  
8477933234|pankajams1234@gmail.com|LinkedIn: Pankajkumardev0 |GitHub: PankajKumardev |Portfolio 
PROFILE SUMMARY    
I’m a Full Stack Developer who loves building web applications that are both functional and easy to use. I work with React.js, 
Next.js, Express.js, and TypeScript and enjoy solving problems through clean, efficient code. I’m also comfortable with tools 
like PostgreSQL, Docker, and CI/CD pipelines, and I’ve contributed to open-source projects to help improve them. 
Experience     
Software Developer (Open Source) –  Hacktoberfest                      
Remote | October 2024 
• Contributed to 5+ open-source projects, fixing bugs and adding new features. 
• Collaborated with developers globally, managing 5+ pull requests and resolving 10+ issues using Git. 
Projects                                                                                                                                                                      
FlowPay |Code| Link                                                                                                                                                     
November 2024  
Tools: Next.js, TailwindCSS, Turborepo, Docker, AWS EC2 (CI/CD), PostgreSQL, Prisma, NextAuth  
• Built a simulated banking system for peer-to-peer payments, designed to feel like real-world bank transfers (e.g., 
HDFC, Axis). 
• Designed a responsive user interface with Next.js and TailwindCSS, ensuring seamless usability across devices. 
• Set up a monorepo structure with Turborepo and Automated deployments using CI/CD pipelines on AWS EC2, 
reducing deployment time by 40%. 
Feed-Wall |Code| Link                                                                  
Tools: Next.js, TailwindCSS, Nextauth, Gemini LLM,  Prisma, PostgreSQL, Docker 
January 2025  
• Developed a feedback widget that can be embedded into websites, used by 10+ sites to collect user feedback. 
• Used TypeScript and Zod to catch errors early in development, reducing bugs by 30%. 
• Added AI-powered summaries to help users quickly understand feedback, saving them 2+ hours/week. 
Ui-Unify |Code| Link                                                                                                                                                                   
Tools: Nextjs, TailwindCSS, Framer Motion, Gemini LLM 
December 2024  
• Created a unified UI component library with 30+ components from popular libraries like Aceternity UI, Magic UI, and 
Shadcn/ui. 
• Gemini LLM integration providing AI-powered component generation and  installation guides for every library. 
• Provided comprehensive learning resources, including tutorials and legal pages (Privacy Policy, Terms of Service). 
Technical Skills    
• Programming Languages: JavaScript, TypeScript, C++,Python 
• Frontend Development: React.js, Next.js, HTML, CSS, TailwindCSS 
• Backend Development: Node.js, Express.js, REST APIs, Prisma 
• DevOps Tools: Docker, CI/CD  (AWS EC2) 
• Database Management: MongoDB, PostgreSQL 
• Tools & Technologies: Postman, Git ,AWS, Visual Studio Code (VS Code) 
EDUCATION                                                                                                                                                              
• Vivekananda Institute of Professional Studies                                                                 
• Bachelor of Computer Applications, CGPA: 8.2                                                   - New Delhi  
Expected Graduation, July 2026 
ACHIEVEMENTS                                                                                                                                                       
• Open Source Contributions: Improved functionality and code quality for 5+ projects during Hacktoberfest. 
• ACE Society: Collaborated on 5+ web development projects. `}]`;
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
