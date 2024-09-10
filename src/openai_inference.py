import os, sys
import json
import tiktoken
from openai import OpenAI
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional

sys.path.append(sys.path[0])
from src.utilities import *

class Inference:
    def __init__(self,
                 agent):
        self.agent = agent
        
        ### MODEL SELECTION
        self.primary_model = "gpt-4o-2024-08-06"
        self.secondary_model = "gpt-4o-mini-2024-07-18"
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dimensions = 512

        ### CLASS IMPORTS
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        ### KNOWLEDGE GRAPH PARAMETERS
        self.graph_template_location = f"./prompt-templates/knowledge-graph-generation-prompt.txt"
        self.summary_template_location = f"./prompt-templates/summary-generation-prompt.txt"
        self.allowed_node_types = ["Technology", "Concept", "Image", "Organization", "Process", "Person", "Group", "Experiment", "Experience", "Location", "Emotion", "Anatomy", "Organism", "Medicine", "Object", "Practice"]

    def get_token_len(self,
                      text):
        return len(self.tokenizer.encode(text))

    async def spin_network(self,
                           message_thread):
        """
        FORMULATE NETWORK OF MESSAGE THREAD FOR STRUCTURED ABSTRACTIONS OF THE CONVERSATION
        """
        timer = Timer()
        
        ### FIRST COMPRESS THE MESSAGE THREAD FOR REDUCED COMPUTATIONAL LOAD OF KG
        # compressed_message_thread, compression_cost = await self.compress_message_thread(message_thread['text'])
        compressed_message_thread = message_thread['text']
        compression_cost = 0
        # print(message_thread['text'])
        # print('\n---------\n')
        # print(compressed_message_thread)
        
        ### STRUCTURE THE SYSTEM PROMPT FOR KNOWLEDGE GRAPH GENERATION
        with open(self.graph_template_location, 'r') as file:
            knowledge_graph_prompt_template = file.read()
        knowledge_graph_prompt = knowledge_graph_prompt_template.format(
            node_types=self.allowed_node_types
        )
    
        ### OPENAI INFERENCE
        input_messages = [
            {"role": "system", "content": knowledge_graph_prompt},
            {"role": "user", "content": compressed_message_thread},
        ]
        completion = await self.async_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=input_messages,
            response_format=KnowledgeGraph,
        )
        knowledge_graph_text = completion.choices[0].message.content

        ### CALCULATE COST FOR KG GENERATION
        network_cost = self.openai_cost(
            ingress=input_messages,
            egress=knowledge_graph_text
        )

        ### FIT PYDANTIC CLASS
        raw_knowledge_graph = KnowledgeGraph(**json.loads(knowledge_graph_text))
        knowledge_graph = self.agent.KG.update_knowledge_graph(
            knowledge_graph=raw_knowledge_graph,
            source=message_thread['thread_id'],
            authors=message_thread['users'],
            conversation_title=message_thread['title']
        )
        
        total_cost = compression_cost + network_cost
        time_taken = timer.get_elapsed_time()
        metadata = {
            'cost': total_cost,
            'time_taken': time_taken
        }
        return knowledge_graph, metadata

    async def compress_message_thread(self,
                                      message_thread):
        ### STRUCTURE THE SYSTEM PROMPT FOR THREAD SUMMARIZATION
        with open(self.summary_template_location, 'r') as file:
            summary_prompt_template = file.read()
            
        ### OPENAI INFERENCE
        input_messages = [
            {"role": "system", "content": summary_prompt_template},
            {"role": "user", "content": message_thread},
        ]
        completion = await self.async_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=input_messages
        )

        ### GET COSTS
        response = completion.choices[0].message.content
        compression_cost = self.openai_cost(
            ingress=input_messages,
            egress=response
        )
        return response, compression_cost

    def get_embedding(self,
                      text):
       text = text.replace("\n", " ")
       return self.client.embeddings.create(
           input=[text], 
           model=self.embedding_model, 
           dimensions=self.embedding_dimensions
       ).data[0].embedding

    def openai_cost(self,
                    ingress,
                    egress,
                    model=None,
                    return_egress_tokens=False):
        if not model:
            model = self.primary_model
        
        ### INGRESS IS MESSAGE LIST / EGRESS IS RESPONSE STRING
        if isinstance(ingress, list):
            ingress_tokens = 0
            for message in ingress:
                if message['content']:
                    message_tokens = len(self.tokenizer.encode(message.get('content')))
                    ingress_tokens += message_tokens
        elif isinstance(ingress, str):
            ingress_tokens = len(self.tokenizer.encode(ingress))
        elif isinstance(ingress, int):
            ingress_tokens = ingress
        if isinstance(egress, int):
            egress_tokens = egress
        else:
            egress_tokens = len(self.tokenizer.encode(egress))
                            
        if model in ["gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-vision-preview"]:
            prompt_cost = (ingress_tokens / 1000)*0.01
            response_cost = (egress_tokens / 1000)*0.03

        elif model in ["gpt-4o-mini-2024-07-18"]:
            prompt_cost = (ingress_tokens / 1000000)*0.15
            response_cost = (egress_tokens / 1000000)*0.60

        elif model in ["gpt-4o-2024-05-13"]:
            prompt_cost = (ingress_tokens / 1000000)*5.00
            response_cost = (egress_tokens / 1000000)*15.00

        elif model in ["gpt-4-turbo-2024-04-09"]:
            prompt_cost = (ingress_tokens / 1000000)*10.00
            response_cost = (egress_tokens / 1000000)*30.00
            
        elif model in ["gpt-4o-2024-08-06"]:
            prompt_cost = (ingress_tokens / 1000000)*2.50
            response_cost = (egress_tokens / 1000000)*10.00
            
        elif model in ["gpt-4"]:
            prompt_cost = (egress_tokens / 1000)*0.03
            response_cost = (egress_tokens / 1000)*0.06
            
        elif model in ["gpt-4"]:
            prompt_cost = (egress_tokens / 1000)*0.03
            response_cost = (egress_tokens / 1000)*0.06

        elif model in ["gpt-4-32k"]:
            prompt_cost = (egress_tokens / 1000)*0.06
            response_cost = (egress_tokens / 1000)*0.12

        elif model in ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"]:
            prompt_cost = (egress_tokens / 1000)*0.0010
            response_cost = (egress_tokens / 1000)*0.0020

        elif model in ["gpt-3.5-turbo-instruct"]:
            prompt_cost = (egress_tokens / 1000)*0.0015
            response_cost = (egress_tokens / 1000)*0.0020

        if not return_egress_tokens:
            return prompt_cost+response_cost     
        else:
            return prompt_cost+response_cost, egress_tokens

#########################
### CLASS DEFINITIONS ###
#########################
class KnowledgeGraph(BaseModel):
    
    class Node(BaseModel):
        id: str = Field(..., description="textual id for the node")
        type: str = Field(..., description="explicit type for the node")
        description: str = Field(..., description="detailed description of particular properties or definitions related to the node")
        embedding: Optional[List[float]] = None
        source: Optional[List[str]] = None
        authors: Optional[List[str]] = None
        conversation_title: Optional[List[str]] = None
        
    class Relationship(BaseModel):
        source: str = Field(..., description="textual id for the source node")
        target: str = Field(..., description="explicit type for the target node")
        type: str = Field(..., description="explicit type for the relationship")
        
    nodes: list[Node]
    rels: list[Relationship]