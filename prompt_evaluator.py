import asyncio
import pandas as pd
from vertexai.generative_models import GenerativeModel
from tqdm.asyncio import tqdm_asyncio
import backoff

class ReviewModelError(Exception):
    """Custom exception for review model errors."""
    pass

class PromptEvaluator:
    def __init__(self, df_train, target_model_name, target_model_config, review_model_name, review_model_config, safety_settings, review_prompt_template_path):
        self.df_train = df_train
        self.target_model_name = target_model_name
        self.target_model_config = target_model_config
        self.review_model_name = review_model_name
        self.review_model_config = review_model_config
        self.safety_settings = safety_settings
        self.review_prompt_template_path = review_prompt_template_path

        self.target_model = GenerativeModel(self.target_model_name)
        self.review_model = GenerativeModel(self.review_model_name)

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def generate_target_model_response(self, question, prompt):
        target_model = GenerativeModel(
            self.target_model_name,
            generation_config=self.target_model_config,
            safety_settings=self.safety_settings,
            system_instruction=prompt
        )

        response = await target_model.generate_content_async(
            question,
            stream=False,
        )
        return response.text

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def generate_review_model_response(self, review_prompt):
        review_response = await self.review_model.generate_content_async(
            [review_prompt],
            generation_config=self.review_model_config,
            safety_settings=self.safety_settings,
            stream=False,
        )
        return review_response.text.strip().lower()

    async def generate_and_review(self, row, prompt):
        try:
            model_response = await self.generate_target_model_response(row["question"], prompt)

            # Load the review prompt from the text file
            with open(self.review_prompt_template_path, 'r') as f:
                review_prompt_template = f.read().strip()

            # Fill in the review prompt with the model response and ground truth
            review_prompt = review_prompt_template.format(model_response=model_response, ground_truth=row['answer'])

            # Now use the review model to compare the model response with the ground truth
            review_result = await self.generate_review_model_response(review_prompt)

            # Check if the target model returned a valid response
            if not model_response or not isinstance(model_response, str):
                raise ReviewModelError("Target model did not return a valid response.")

            # Assert that the review model returns either 'true' or 'false'
            if review_result not in ['true', 'false']:
                raise ReviewModelError("Review model did not return a valid response.")

            is_correct = review_result == 'true'  # Check if the response is 'True'

            return row.name, model_response, is_correct 
        except ReviewModelError as e:
            print(f"Error: {e}. The review model did not return a valid response. Terminating the program.")
            raise  # Re-raise the exception to be caught in the main function
        except Exception as e:
            print(f"An error occurred: {e}. Terminating the program.")
            raise  # Re-raise the exception to be caught in the main function

    async def evaluate_prompt(self, prompt):
        tasks = [self.generate_and_review(row, prompt) for _, row in self.df_train.iterrows()]

        # Create a tqdm progress bar
        with tqdm_asyncio(total=len(tasks), desc="Evaluating Prompt") as pbar:

            async def wrapped_task(task):
                result = await task
                pbar.update(1)  # Update progress bar after task completion
                return result

            # Run tasks with progress bar updates
            results = await asyncio.gather(*[wrapped_task(task) for task in tasks])

        # Prepare results for saving
        evaluation_results = []
        for index, model_response, is_correct in results:
            if index is not None:  # Check if the index is valid
                self.df_train.loc[index, 'model_response'] = model_response
                self.df_train.loc[index, 'is_correct'] = is_correct
                evaluation_results.append({
                    'question': self.df_train.loc[index, 'question'],
                    'ground_truth': self.df_train.loc[index, 'answer'],
                    'model_response': model_response,
                    'is_correct': is_correct
                })

        overall_accuracy = sum(self.df_train["is_correct"]) / len(self.df_train)

        # Save results to CSV
        results_df = pd.DataFrame(evaluation_results)
        results_csv_path = 'evaluation_results.csv'
        results_df.to_csv(results_csv_path, index=False)

        return overall_accuracy

    async def main(self, prompt):
        try:
            accuracy = await self.evaluate_prompt(prompt)
            print(f"Overall accuracy for the prompt: {accuracy:.2f}")
        except ReviewModelError:
            print("The program has terminated due to an invalid response from the review model.")
        except Exception as e:
            print(f"The program has terminated due to an unexpected error: {e}")