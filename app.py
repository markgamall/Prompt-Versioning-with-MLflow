import os
import warnings
import pandas as pd
import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import evaluate

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("API key not found in environment variables")

# Set the tracking URI to a local directory
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def log_llm_model_and_evaluate(
    experiment_name, model_name, temperature, max_tokens,
    system_message_template, input_text, tag_name, run_name, eval_df
):
    # Define the MLflow model class
    class SummarizationModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.llm = ChatGroq(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=groq_api_key
            )
        
        def predict(self, context, model_input):
            system_message = system_message_template.format(model_input=model_input)
            prompt_template = ChatPromptTemplate.from_template(system_message)
            patient_messages = prompt_template.format_messages(text=model_input)
            response = self.llm(patient_messages)
            return response.content

    # Set the experiment name
    mlflow.set_experiment(experiment_name)

    # Start an MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("model", model_name)
        mlflow.log_param("temperature", temperature)
        mlflow.log_param("max_tokens", max_tokens)

        # Create and log the system message
        system_message = system_message_template.format(model_input=input_text)
        mlflow.log_text(system_message, "prompt.txt")

        # Log the model
        mlflow.pyfunc.log_model(
            artifact_path="summarization_model",
            python_model=SummarizationModel(),
            conda_env=None,
        )

        # Log the run tag
        mlflow.set_tag("run_name", tag_name)
        
        # Load the model using MLflow
        model_uri = f"runs:/{run.info.run_id}/summarization_model"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # Evaluate the model using ROUGE
        rouge = evaluate.load('rouge')
        predictions = []
        references = []
        inputs = []
        results_list = []
        for _, row in eval_df.iterrows():
            input_text = row["inputs"]
            ground_truth = row["ground_truth"]
            predicted_summary = loaded_model.predict(input_text)
            predictions.append(predicted_summary)
            references.append(ground_truth)
            inputs.append(input_text)
            
            # Append the results for DataFrame creation
            results_list.append({
                "inputs": input_text,
                "predicted_summary": predicted_summary,
                "ground_truth": ground_truth
            })
        
        # Compute ROUGE scores
        rouge_results = rouge.compute(predictions=predictions, references=references)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results_list)
        eval_results_table = results_df.to_string(index=False)
        mlflow.log_text(eval_results_table, "eval_results_table.txt")
        
        # Log ROUGE metrics
        for metric, score in rouge_results.items():
            mlflow.log_metric(metric, score)
        
        print(f"Model and evaluation results logged in run: {run.info.run_id}")
        print("ROUGE Scores:", rouge_results)

# Example usage
def main():
    eval_df = pd.DataFrame({
        "inputs": [
            "The X-ray of the lumbar spine reveals mild degenerative disc disease with reduced disc height at L4-L5. There is evidence of osteophyte formation at L3-L4 and L4-L5. No acute fractures are observed. The vertebral alignment is normal, and there is no significant scoliosis or kyphosis. The sacroiliac joints appear intact with no signs of inflammatory changes.",           
            "The chest X-ray shows a normal heart size and clear lung fields with no signs of pneumonia, fractures, or pleural effusion. The mediastinum is within normal limits. The rib cage is intact, and there are no abnormal masses or calcifications observed. The aorta appears normal in caliber and there are no signs of pulmonary vascular congestion",
            "The X-ray of the left lung shows a slight increase in density at the lung base, which could be indicative of atelectasis. There are no signs of pleural effusion or pulmonary nodules. The right lung fields are clear and there is no evidence of a lung mass. The heart silhouette is normal, and the diaphragm appears intact with no abnormal elevation.",
            "The X-ray of the chest reveals a small calcified granuloma in the right upper lobe, likely a benign finding. There is no evidence of significant cardiomegaly or other abnormal findings in the heart. The rib cage is intact and the lung fields are clear of any acute pathology. The trachea is midline, and the vascular markings are within normal limits."
        ],
        "ground_truth": [
            "The lumbar spine X-ray shows mild degenerative changes with reduced disc height at L4-L5 and osteophyte formation at L3-L4 and L4-L5. No acute fractures or significant alignment issues are present.",
            "The chest X-ray is normal with clear lung fields, no signs of pneumonia or fractures, and a normal heart size. There are no abnormalities in the mediastinum or rib cage.",
            "The X-ray indicates slight atelectasis in the left lung base with no pleural effusion or lung masses. The right lung and heart appear normal, and the diaphragm is intact.",
            "The chest X-ray reveals a small benign granuloma in the right upper lobe, with no significant cardiomegaly or other abnormalities. The rib cage and lung fields are clear.",
        ],
    })

    system_message_template = """
    Summarize the following X-ray report into a concise summary that captures the essential information. Ensure that the summary is clear and to the point.

    X-ray Report: ```{model_input}```

    """

    log_llm_model_and_evaluate(
        experiment_name="comparing llms",
        model_name="llama-3.1-70b-versatile",
        temperature=0.5,
        max_tokens=1000,
        system_message_template=system_message_template,
        input_text="Sample input text",
        tag_name="first llm model - x-ray summerizer",
        run_name="Report Summerizer 2",
        eval_df=eval_df
    )

if __name__ == "__main__":
    main()
