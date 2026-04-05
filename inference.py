import os
import json
import asyncio
from openai import AsyncOpenAI
from env import DataJanitorEnv
from models import DataJanitorAction, ActionType


SYSTEM_PROMPT = """
You are a data engineering agent. Your task is to clean data and save results to SQLite databases.

Available actions:
- list_files: List files in workspace
- read_file: Read a file (provide file_path)
- run_python: Execute Python code (provide python_code)

Respond with ONLY valid JSON in this format:
{"action_type": "run_python|list_files|read_file", "python_code": "...", "file_path": "..."}

Do not include any other text or explanations.
"""

class DataEngineerAgent:
    def __init__(self, task_level: str):
        self.task_level = task_level
        self.env = DataJanitorEnv(task_level, max_steps=10)
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def run(self):
        print(f"[START] {self.task_level.upper()} TASK (GPT-4)")
        
        response = self.env.reset()
        step_num = 0
        
        while not response.done:
            step_num += 1
            
            # Add current observation to conversation
            obs_text = f"""
Task: {response.observation.task_description}
Files: {', '.join(response.observation.files_in_workspace)}
Database: {response.observation.database_info}
Stdout: {response.observation.stdout}
Stderr: {response.observation.stderr}
"""
            self.conversation.append({"role": "user", "content": obs_text})
            
            # Get action from LLM
            try:
                llm_response = await self.client.chat.completions.create(
                    model="gpt-4",
                    messages=self.conversation,
                    max_tokens=500,
                    temperature=0.1
                )
                
                action_json = llm_response.choices[0].message.content.strip()
                action_data = json.loads(action_json)
                
                action = DataJanitorAction(**action_data)
                
                print(f"[STEP {step_num}] Action: {action.action_type.value}")
                
                # Execute action
                response = self.env.step(action)
                
                # Add result to conversation
                result_text = f"Reward: {response.reward}, Done: {response.done}"
                self.conversation.append({"role": "assistant", "content": result_text})
                
                if response.reward > 0:
                    print(f"Reward: +{response.reward:.2f}")
                    
            except Exception as e:
                print(f"Error in step {step_num}: {e}")
                break
        
        final_score = self.env._get_current_score()
        print(f"[END] Task completed! Final Score: {final_score:.2f}")

    async def cleanup(self):
        self.env.cleanup()


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    for task_level in ["easy", "medium", "hard"]:
        agent = DataEngineerAgent(task_level)
        try:
            await agent.run()
        finally:
            await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())