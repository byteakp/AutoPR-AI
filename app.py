import os
import subprocess
import shutil
import logging
import aiohttp
import asyncio
import threading
from typing import List, Dict, Tuple, Union
from github import Github, GithubException
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Main Agent Class ---
class CodeRefactorAgent:
    """
    An agent that automatically refactors Python code in a GitHub repository,
    identifies code smells, and creates a pull request with the improvements.
    """
    def __init__(self, repo_url: str, github_token: str, gemini_api_key: str):
        if not all([repo_url, github_token, gemini_api_key]):
            raise ValueError("Repo URL, GitHub token, and Gemini API key are required.")

        self.repo_url = repo_url
        self.github_token = github_token
        self.gemini_api_key = gemini_api_key

        self.repo_name = self.repo_url.split('/')[-1].replace('.git', '')
        self.repo_owner = self.repo_url.split('/')[-2]
        self.clone_path = os.path.join(os.getcwd(), self.repo_name)
        self.github_client = Github(github_token)

    def clone_repository(self) -> None:
        """Clones the specified GitHub repository to a local directory."""
        if os.path.exists(self.clone_path):
            logging.info(f"Repository already exists at {self.clone_path}. Removing it.")
            shutil.rmtree(self.clone_path)

        try:
            logging.info(f"Cloning repository: {self.repo_url} into {self.clone_path}")
            subprocess.run(['git', 'clone', self.repo_url, self.clone_path], check=True, capture_output=True, text=True)
            logging.info("Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to clone repository: {e.stderr}")
            raise

    def find_python_files(self) -> List[str]:
        """Finds all Python files in the cloned repository, ignoring virtual environments."""
        python_files = []
        for root, dirs, files in os.walk(self.clone_path):
            dirs[:] = [d for d in dirs if d not in ['venv', '.venv', 'env', '.git']]
            for file in files:
                if file.endswith('.py'):
                    relative_path = os.path.relpath(os.path.join(root, file), self.clone_path)
                    python_files.append(relative_path)
        logging.info(f"Found {len(python_files)} Python files to analyze.")
        return python_files

    def analyze_for_code_smells(self, file_path: str, min_lines: int = 50) -> bool:
        """A simple code smell detector. Checks if a file is longer than min_lines."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            
            if line_count > min_lines:
                logging.info(f"Code smell detected in {os.path.basename(file_path)}: File is long ({line_count} lines).")
                return True
            return False
        except Exception as e:
            logging.warning(f"Could not analyze file {file_path}: {e}")
            return False

    async def refactor_file_content(self, file_content: str) -> str:
        """Refactors the content of a Python file using the Gemini Pro API."""
        prompt = (
            "Please refactor the following Python code to be more readable, efficient, and adhere to PEP 8 standards.\n"
            "Add docstrings to all functions and classes. Add comments explaining complex logic.\n"
            "Ensure the refactored code maintains the exact original functionality.\n"
            "Only return the raw refactored Python code, without any markdown formatting or explanations.\n\n"
            "Original Code:\n"
            "```python\n"
            f"{file_content}\n"
            "```\n\n"
            "Refactored Code:"
        )
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        headers = {'Content-Type': 'application/json'}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(api_url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            if not result.get('candidates'):
                                logging.warning("Gemini API returned no candidates.")
                                return file_content
                            refactored_code = result['candidates'][0]['content']['parts'][0]['text']
                            if refactored_code.strip().startswith("```python"):
                                refactored_code = refactored_code.strip()[9:]
                            if refactored_code.strip().endswith("```"):
                                refactored_code = refactored_code.strip()[:-3]
                            return refactored_code.strip()
                        else:
                            error_text = await response.text()
                            logging.error(f"Gemini API error: {response.status} - {error_text}")
            except Exception as e:
                logging.error(f"Error calling Gemini API (attempt {attempt + 1}): {e}")
                await asyncio.sleep(2 ** attempt)
        
        logging.warning("Failed to refactor code after multiple attempts. Returning original code.")
        return file_content

    def create_pull_request(self, refactored_files: Dict[str, str]) -> None:
        """Creates a new branch, commits the refactored files, and opens a pull request."""
        if not refactored_files:
            logging.info("No files were refactored. Skipping pull request creation.")
            return

        try:
            repo = self.github_client.get_repo(f"{self.repo_owner}/{self.repo_name}")
            default_branch = repo.get_branch(repo.default_branch)
            new_branch_name = "ai-refactor-suggestions"
            
            try:
                ref = repo.get_git_ref(f"heads/{new_branch_name}")
                ref.delete()
                logging.info(f"Deleted existing branch: {new_branch_name}")
            except GithubException as e:
                if e.status != 404: raise

            repo.create_git_ref(ref=f'refs/heads/{new_branch_name}', sha=default_branch.commit.sha)
            logging.info(f"Created new branch: {new_branch_name}")

            for file_path, content in refactored_files.items():
                repo.update_file(
                    path=file_path,
                    message=f"feat: AI-powered refactoring for {os.path.basename(file_path)}",
                    content=content,
                    sha=repo.get_contents(file_path, ref=new_branch_name).sha,
                    branch=new_branch_name
                )
                logging.info(f"Updated file {file_path} in branch {new_branch_name}")

            pr_title = "🤖 AI Refactoring Suggestions"
            pr_body = (
                "Hello!\n\n"
                "I'm an automated refactoring agent. I've analyzed the codebase and "
                "identified some areas that could be improved for readability and maintainability.\n\n"
                "This pull request contains the suggested changes. Please review them carefully."
            )
            
            pr = repo.create_pull(title=pr_title, body=pr_body, head=new_branch_name, base=repo.default_branch)
            logging.info(f"Successfully created pull request: {pr.html_url}")

        except GithubException as e:
            logging.error(f"GitHub API Error: {e.status} - {e.data}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during pull request creation: {e}")

    def cleanup(self) -> None:
        """Removes the cloned repository directory."""
        if os.path.exists(self.clone_path):
            try:
                shutil.rmtree(self.clone_path)
                logging.info(f"Cleaned up local repository at {self.clone_path}")
            except OSError as e:
                logging.error(f"Error cleaning up repository: {e}")

    async def run(self) -> None:
        """Runs the entire refactoring and pull request workflow."""
        try:
            self.clone_repository()
            python_files = self.find_python_files()
            refactored_files = {}

            tasks = []
            for file_path in python_files:
                absolute_path = os.path.join(self.clone_path, file_path)
                if self.analyze_for_code_smells(absolute_path):
                    tasks.append(self.process_file(file_path, absolute_path))

            results = await asyncio.gather(*tasks)
            for file_path, refactored_content in results:
                if refactored_content:
                    refactored_files[file_path] = refactored_content
            
            if refactored_files:
                self.create_pull_request(refactored_files)

        finally:
            self.cleanup()

    async def process_file(self, file_path: str, absolute_path: str) -> Tuple[str, Union[str, None]]:
        """Reads, refactors, and returns the content of a single file."""
        logging.info(f"Processing {file_path} for refactoring.")
        try:
            with open(absolute_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            refactored_content = await self.refactor_file_content(original_content)
            
            if refactored_content != original_content:
                logging.info(f"Successfully refactored {file_path}.")
                return file_path, refactored_content
            else:
                logging.info(f"No changes made to {file_path} after refactoring attempt.")
                return file_path, None
                
        except Exception as e:
            logging.error(f"Failed to process file {file_path}: {e}")
            return file_path, None

# --- Flask App ---
app = Flask(__name__)

@app.route('/refactor', methods=['POST'])
def trigger_refactor():
    data = request.get_json()
    if not data or 'repo_url' not in data:
        return jsonify({"error": "repo_url is required in JSON payload"}), 400
    
    repo_url = data['repo_url']
    github_token = os.environ.get("GITHUB_TOKEN")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    if not github_token or not gemini_api_key:
        return jsonify({"error": "API keys are not configured on the server"}), 500

    try:
        # Define the async function that the agent will run
        async def run_agent_async():
            agent = CodeRefactorAgent(
                repo_url=repo_url,
                github_token=github_token,
                gemini_api_key=gemini_api_key
            )
            await agent.run()

        # This function will run in a separate thread
        def run_in_thread():
            # Create and set a new event loop for the new thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Run the async agent function until it completes
            loop.run_until_complete(run_agent_async())
            loop.close()

        # Create and start the background thread
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        
        return jsonify({"status": f"Refactoring process started for {repo_url}"}), 202
    except Exception as e:
        logging.error(f"Failed to start refactoring process: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
