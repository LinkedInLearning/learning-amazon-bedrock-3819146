#Imports
import boto3
import json

#Create the client
client = boto3.client(service_name='bedrock-runtime')

#Construct the body
#specify your prompt
body = json.dumps({
    "prompt": "", 
    "maxTokens": 200,
    "temperature": 0.5,
    "topP": 0.5
})

#Specify model id and content types
modelId = ''
accept = 'application/json'
contentType = 'application/json'

#Invoke the model
response = client.invoke_model(
    body=body, 
    modelId=modelId, 
    accept=accept, 
    contentType=contentType
)

#Extract the response
response_body = json.loads(response.get('body').read())

#Display the output
print(response_body.get('completions')[0].get('data').get('text'))