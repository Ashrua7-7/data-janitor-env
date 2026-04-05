import asyncio
import sqlite3
from env import DataJanitorEnv
from models import DataJanitorAction, ActionType


class MockAgent:
    def __init__(self, task_level: str):
        self.task_level = task_level
        self.env = DataJanitorEnv(task_level, max_steps=10)

    async def run(self):
        print(f"[START] {self.task_level.upper()} TASK")
        
        response = self.env.reset()
        step_num = 0
        
        while not response.done:
            step_num += 1
            action = self._get_action(step_num, response.observation)
            
            if action is None:
                break
            
            print(f"[STEP {step_num}] Action: {action.action_type.value}")
            if action.action_type == ActionType.RUN_PYTHON:
                print("Code executed successfully")
            
            response = self.env.step(action)
            
            if response.reward > 0:
                print(f"Reward: +{response.reward:.2f}")
        
        print(f"[END] Task completed! Final Score: {self.env._get_current_score():.2f}")

    def _get_action(self, step_num: int, obs) -> DataJanitorAction | None:
        if self.task_level == "easy":
            return self._easy_step(step_num, obs)
        elif self.task_level == "medium":
            return self._medium_step(step_num, obs)
        elif self.task_level == "hard":
            return self._hard_step(step_num, obs)
        return None

    def _easy_step(self, step_num: int, obs) -> DataJanitorAction | None:
        if step_num == 1:
            return DataJanitorAction(action_type=ActionType.LIST_FILES)
        elif step_num == 2:
            code = """
import pandas as pd
import sqlite3

df = pd.read_csv('users.csv')
conn = sqlite3.connect('output.db')
df.to_sql('users', conn, if_exists='replace', index=False)
conn.close()
"""
            return DataJanitorAction(action_type=ActionType.RUN_PYTHON, python_code=code.strip())
        else:
            return None

    def _medium_step(self, step_num: int, obs) -> DataJanitorAction | None:
        if step_num == 1:
            return DataJanitorAction(action_type=ActionType.LIST_FILES)
        elif step_num == 2:
            code = """
import pandas as pd
import sqlite3

df = pd.read_csv('sales.csv')
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False).dt.strftime('%Y-%m-%d')
conn = sqlite3.connect('output.db')
df.to_sql('clean_sales', conn, if_exists='replace', index=False)
conn.close()
"""
            return DataJanitorAction(action_type=ActionType.RUN_PYTHON, python_code=code.strip())
        else:
            return None

    def _hard_step(self, step_num: int, obs) -> DataJanitorAction | None:
        if step_num == 1:
            return DataJanitorAction(action_type=ActionType.LIST_FILES)
        elif step_num == 2:
            code = """
import pandas as pd
import json
import sqlite3

with open('users.json', 'r') as f:
    users = json.load(f)
users_df = pd.DataFrame(users)

purchases_df = pd.read_csv('purchases.csv')

# Filter users who opted in
opted_in_users = users_df[users_df['marketing_opt_in'] == True]

# Merge and calculate LTV
merged = pd.merge(purchases_df, opted_in_users, on='user_id')
ltv = merged.groupby('user_id')['amount'].sum().reset_index()
ltv.columns = ['user_id', 'total_ltv']

conn = sqlite3.connect('output.db')
ltv.to_sql('ltv_report', conn, if_exists='replace', index=False)
conn.close()
"""
            return DataJanitorAction(action_type=ActionType.RUN_PYTHON, python_code=code.strip())
        else:
            return None


async def main():
    for task_level in ["easy", "medium", "hard"]:
        agent = MockAgent(task_level)
        try:
            await agent.run()
        finally:
            agent.env.cleanup()


if __name__ == "__main__":
    asyncio.run(main())