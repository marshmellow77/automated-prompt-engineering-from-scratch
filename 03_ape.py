import asyncio
import os
import pandas as pd
from vertexai.generative_models import GenerativeModel, HarmBlockThreshold, HarmCategory
import re
import aiofiles
import datetime
import aioconsole
from prompt_evaluator import PromptEvaluator
import backoff


class APD:
    def __init__(self, num_prompts, starting_prompt, df_train, metaprompt_template_path, generation_model_name, generation_config, safety_settings, target_model_name, target_model_config, review_model_name, review_model_config, review_prompt_template_path):
        self.num_prompts = num_prompts
        self.starting_prompt = starting_prompt
        self.df_train = df_train
        self.metaprompt_template_path = metaprompt_template_path
        self.generation_model_name = generation_model_name
        self.generation_config = generation_config
        self.safety_settings = safety_settings

        # Initialize the generation model
        self.generation_model = GenerativeModel(self.generation_model_name)

        # Create the "runs" folder if it doesn't exist
        self.runs_folder = "runs"
        os.makedirs(self.runs_folder, exist_ok=True)
        
        self.run_folder = self.create_run_folder()
        self.prompt_history = os.path.join(self.run_folder, 'prompt_history.txt')
        self.prompt_history_chronlogical = os.path.join(self.run_folder, 'prompt_history_chronlogical.txt')
        
        # Initialize the PromptEvaluator
        self.prompt_evaluator = PromptEvaluator(
            df_train,
            target_model_name,
            target_model_config,
            review_model_name,
            review_model_config,
            safety_settings,
            review_prompt_template_path
        )

        self.user_feedback = ""
        self.best_prompt = starting_prompt
        self.best_accuracy = 0.0

    def create_run_folder(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(self.runs_folder, f'run_{timestamp}')  # Join with runs_folder
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def create_prompt_subfolder(self, prompt_number):
        prompt_folder = os.path.join(self.run_folder, f'prompt_{prompt_number}')
        os.makedirs(prompt_folder, exist_ok=True)
        return prompt_folder

    def read_and_sort_prompt_accuracies(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        pattern = re.compile(r'<PROMPT>\n<PROMPT_TEXT>\n(.*?)\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: ([0-9.]+)\n</ACCURACY>\n</PROMPT>', re.DOTALL)
        matches = pattern.findall(content)
        
        sorted_prompts = sorted(matches, key=lambda x: float(x[1]))  # Sort in ascending order
        return sorted_prompts

    def write_sorted_prompt_accuracies(self, file_path, sorted_prompts):
        sorted_prompts_string = ""
        with open(file_path, 'w') as f:
            for prompt, accuracy in sorted_prompts:
                s = f"<PROMPT>\n<PROMPT_TEXT>\n{prompt}\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: {accuracy}\n</ACCURACY>\n</PROMPT>\n\n"
                f.write(s)
                sorted_prompts_string += s
                
        return sorted_prompts_string

    def update_metaprompt(self, file_path, metaprompt_template_path):
        sorted_prompts = self.read_and_sort_prompt_accuracies(file_path)
        sorted_prompts_string = self.write_sorted_prompt_accuracies(file_path, sorted_prompts)
                
        with open(metaprompt_template_path, 'r') as f:
            metaprompt_template = f.read()
        
        metaprompt = metaprompt_template.format(prompt_scores=sorted_prompts_string, human_feedback=self.user_feedback)
        
        return metaprompt

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def generate_with_backoff(self, metaprompt):
        response = self.generation_model.generate_content(
            metaprompt,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            stream=False,
        )
        return response

    async def main(self):
        prompt_accuracies = []
        best_prompt = self.starting_prompt
        best_accuracy = 0.0

        for i in range(self.num_prompts + 1):
            await aioconsole.aprint("=" * 150)
            await aioconsole.aprint(f"Prompt number {i}")

            if i == 0:
                new_prompt = self.starting_prompt
                # Evaluate the starting prompt
                accuracy = await self.prompt_evaluator.evaluate_prompt(new_prompt)
                best_accuracy = accuracy
                prompt_accuracies.append((new_prompt, accuracy))
            else:
                metaprompt = self.update_metaprompt(self.prompt_history, self.metaprompt_template_path)
                
                try:
                    response = await self.generate_with_backoff(metaprompt)
                except Exception as e:
                    await aioconsole.aprint(f"Failed to generate content after retries: {e}")
                    continue
                
                await aioconsole.aprint("-" * 150)
                await aioconsole.aprint(response.text)
                await aioconsole.aprint("-" * 150)
                
                match = re.search(r'\[\[(.*?)\]\]', response.text, re.DOTALL)
                if match:
                    new_prompt = match.group(1)
                else:
                    await aioconsole.aprint("No new prompt found")
                    continue
            
            # Create a subfolder for the prompt
            prompt_folder = self.create_prompt_subfolder(i)

            # Save the prompt in a text file within the subfolder
            prompt_file_path = os.path.join(prompt_folder, 'prompt.txt')
            with open(prompt_file_path, 'w') as f:
                f.write(new_prompt)

            # Use the PromptEvaluator to evaluate the new prompt
            accuracy = await self.prompt_evaluator.evaluate_prompt(new_prompt)
            
            if i == 0:
                best_accuracy = starting_accuracy = accuracy
            
            prompt_accuracies.append((new_prompt, accuracy))
            await aioconsole.aprint("-" * 150)
            await aioconsole.aprint(f"Overall accuracy for prompt: {accuracy:.2f}")
            await aioconsole.aprint("=" * 150)

            # Update the best prompt if the current accuracy is higher
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_prompt = new_prompt
            
            # Append to prompt_history.txt
            async with aiofiles.open(self.prompt_history, 'a') as f:
                await f.write(f"<PROMPT>\n<PROMPT_TEXT>\n{new_prompt}\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: {accuracy:.2f}\n</ACCURACY>\n</PROMPT>\n\n")
        
            # Append to prompt_history_chronological.txt with prompt number
            async with aiofiles.open(self.prompt_history_chronlogical, 'a') as f:
                await f.write(f"Prompt number: {i}\nPrompt: {new_prompt}\nAccuracy: {accuracy:.2f}\n\n")
                await f.write("=" * 150 + "\n")
            
            # Save the evaluation results in a CSV file within the subfolder
            csv_file_path = os.path.join(prompt_folder, 'evaluation_results.csv')
            evaluation_results = {
                "question": self.df_train["question"],
                "answer": self.df_train["answer"],
                "model_response": self.df_train["model_response"],
                "is_correct": self.df_train["is_correct"]
            }
            evaluation_df = pd.DataFrame(evaluation_results)
            evaluation_df.to_csv(csv_file_path, index=False)

            # Read, sort, and write the updated prompt accuracies to prompt_history.txt
            sorted_prompts = self.read_and_sort_prompt_accuracies(self.prompt_history)
            self.write_sorted_prompt_accuracies(self.prompt_history, sorted_prompts)

        # Output the final best prompt and improvement in accuracy
        starting_accuracy = prompt_accuracies[0][1]  # Get the accuracy of the first prompt
        improvement = best_accuracy - starting_accuracy
        await aioconsole.aprint("=" * 150)
        await aioconsole.aprint(f"Final best prompt: {best_prompt}")
        await aioconsole.aprint(f"Accuracy of best prompt: {best_accuracy:.2f}")
        await aioconsole.aprint(f"Improvement in accuracy: {improvement:.2f}")

if __name__ == "__main__":
    num_prompts = 5
    starting_prompt = "Solve the given problem about geometric shapes. Think step by step."
    
    df_train = pd.read_csv('train.csv')  # Load your training data

    metaprompt_template_path = 'metaprompt_template.txt'
    generation_model_name = "gemini-1.5-pro"
    generation_config = {
        "temperature": 0.7,
    }
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
    target_model_name = "gemini-1.5-flash"
    target_model_config = {
        "temperature": 0, "max_output_tokens": 1000
    }
    review_model_name = "gemini-1.5-flash" 
    review_model_config = {
        "temperature": 0, "max_output_tokens": 10 
    }
    review_prompt_template_path = 'review_prompt_template.txt'  # Path to the review prompt text file

    apd = APD(
        num_prompts, starting_prompt, df_train, 
        metaprompt_template_path, generation_model_name, generation_config, safety_settings, 
        target_model_name, target_model_config, review_model_name, review_model_config, review_prompt_template_path
    )

    asyncio.run(apd.main())