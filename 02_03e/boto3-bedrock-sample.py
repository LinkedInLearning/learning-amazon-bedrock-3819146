#Imports
import boto3
import json
import os

#outputting variables
print (f'AWS_ACCESS_KEY_ID : {os.environ["AWS_ACCESS_KEY_ID"]}')
print (f'AWS_SECRET_ACCESS_KEY : {os.environ["AWS_SECRET_ACCESS_KEY"]}')
print (f'AWS_DEFAULT_REGION : {os.environ["AWS_DEFAULT_REGION"]}')

#Create the bedrock client
bedrock = boto3.client('bedrock-runtime')

#Setting the prompt
prompt_data = """Command: Write me a blog about coaching employees as a leader.

Blog:
"""

#Model specification
modelId = "amazon.titan-text-express-v1"
accept = "application/json"
contentType = "application/json"

#Configuring parameters to invoke the model
body = json.dumps({
    "inputText": prompt_data, 
    "textGenerationConfig": {
         "maxTokenCount" : 1000
    }
})

#Invoke the model
response = bedrock.invoke_model(
    body=body, modelId=modelId, accept=accept, contentType=contentType
)

#Parsing and displaying the output
response_body = json.loads(response.get('body').read())
output = response_body.get('results')[0].get("outputText")
print(output)
