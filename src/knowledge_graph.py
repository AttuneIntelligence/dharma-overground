import json
import os
import sys
import pickle
import openai  # for handling rate limit errors
from typing import List
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import umap
import leidenalg
import igraph as ig
import colorsys
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

sys.path.append(sys.path[0])
from src.openai_inference import *
from src.utilities import *

class NetWeaver:
    def __init__(self, agent):
        self.agent = agent
        self.graph_directory = "./knowledge_graphs"
        
        ### NETWORK GENERATION HYPERPARAMETERS
        self.concurrency_limit = 2
        self.rate_limit_max_retries = 3
        self.rate_limit_restart_time = 10

    def merge_networks(self,
                       list_of_networks):
        """
        TAKE A LIST OF MESSAGE THREAD NETWORKS AND WEAVE THE META-NETWORK
        """
        def node_exists(node_id, combined_graph):
            for existing_node in combined_graph.nodes:
                if existing_node.id == node_id:
                    return existing_node
            return None
        def relationship_exists(source, target, rel_type):
            return any(r.source == source and r.target == target and r.type == rel_type for r in combined_graph.rels)
    
        def compute_midpoint(vector1, 
                             vector2):
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            midpoint = (v1 + v2) / 2
            return midpoint
            
        ### LOOP THROUGH ALL NETWORKS AND ADD NODES + RELATIONSHIPS
        combined_graph = KnowledgeGraph(nodes=[], rels=[])
        replicated_nodes = set()
        for network in tqdm(list_of_networks):
            for node in network.nodes:
                existing_node = node_exists(node.id, combined_graph)
                if existing_node is None:
                    combined_graph.nodes.append(node)
                else:
                    replicated_nodes.add(node.id)
                    ### REMOVE THE EXISTING NODE AND ADD NEW ONE
                    combined_graph.nodes = [n for n in combined_graph.nodes if n.id != node.id]
                    aggregate_node = {
                        'id': node.id,
                        'type': node.type,
                        'description': f"{existing_node.description}\n\n{node.description}",
                        'embedding': compute_midpoint(node.embedding, existing_node.embedding),
                        'source': existing_node.source + node.source,
                        'authors': list(set(existing_node.authors + node.authors)),
                        'conversation_title': existing_node.conversation_title + node.conversation_title
                    }
                    # print(aggregate_node)
                    combined_graph.nodes.append(KnowledgeGraph.Node(**aggregate_node))
    
            for rel in network.rels:
                if not relationship_exists(rel.source, rel.target, rel.type):
                    if not node_exists(rel.source, combined_graph):
                        continue
                    if not node_exists(rel.target, combined_graph):
                        continue
                    combined_graph.rels.append(rel)

        print(f"Meta-network generation completed!")
        print(f"    ==> n total nodes: {len(combined_graph.nodes)}")
        print(f"    ==> replicated Nodes: {list(replicated_nodes)}")
        return combined_graph
    
    async def weave_meta_net(self, message_threads, network_name=None):
        """
        TAKE THE LIST OF MESSAGE THREADS AND CONCURRENTLY GENERATE NETWORKS
        """
        async def process_thread(dharma_message):
            for attempt in range(self.rate_limit_max_retries):
                try:
                    knowledge_graph, kg_metadata = await self.agent.OAI.spin_network(dharma_message)
                    return knowledge_graph, kg_metadata['cost']
                except openai.RateLimitError as e:
                    if attempt == self.rate_limit_max_retries - 1:
                        raise  ### RE-RAISE, EXHAUSTED ALL RETRIES
                    delay = self.rate_limit_restart_time * (2 ** attempt) + random.uniform(0, self.rate_limit_restart_time)
                    print(f"Rate limit reached. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
    
        async def bounded_process_thread(semaphore, dharma_message):
            async with semaphore:
                return await process_thread(dharma_message)
    
        timer = Timer()
    
        ### PARALLEL GENERATION OF KNOWLEDGE GRAPHS
        print(f"Concurrently generating networks...")
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        tasks = [bounded_process_thread(semaphore, message) for message in message_threads.values()]
        
        # Use gather with return_exceptions=True to prevent one failure from stopping all tasks
        results = await async_tqdm.gather(*tasks, desc="Processing threads")
    
        ### PROCESS RESULTS
        all_networks = []
        total_cost = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"An error occurred: {result}")
            else:
                graph, cost = result
                all_networks.append(graph)
                total_cost += cost
    
        ### MERGE NETWORKS INTO A SINGLE META-NETWORK
        print(f"Merging thread networks into single meta-network...")
        meta_network = self.merge_networks(all_networks)
        
        ### SAVE TO FILE AND RETURN
        self.save_network_to_pkl(meta_network, network_name=network_name)
        time_taken = timer.get_elapsed_time()
        print(f"Time taken: {time_taken} seconds")
        
        return meta_network, total_cost

    def select_random_threads(self,
                              message_threads,
                              subset_size=4):
        subset_size = min(subset_size, len(message_threads))
        random_keys = random.sample(list(message_threads.keys()), subset_size)
        subset = {key: message_threads[key] for key in random_keys}
        return subset
    
    def update_knowledge_graph(self, 
                               knowledge_graph, 
                               source, 
                               authors, 
                               conversation_title):
        updated_nodes = []

        for node in knowledge_graph.nodes:
            ### FORCE UPDATE PYDANTIC NODE FIELDS
            node.source = [str(source)]
            node.conversation_title = [conversation_title]
            node.authors = [str(author) for author in authors.copy()] 

            ### COMPUTE EMBEDDING FOR THE EACH NODE
            node.embedding = self.agent.OAI.get_embedding(
                f"{node.id} {node.description}"
            )
            updated_nodes.append(node)

        updated_knowledge_graph = KnowledgeGraph(
            nodes=updated_nodes, rels=knowledge_graph.rels
        )
        return updated_knowledge_graph

    def generate_knowledge_graph_html(self, 
                                      knowledge_graph, 
                                      clustering_resolution=0.3,
                                      options=None):
        if isinstance(knowledge_graph, str):
            knowledge_graph = self.read_network_from_pickle(knowledge_graph)
        
        if options is None:
            options = {}
    
        default_options = {
            "node_color": "#5C88DA",
            "edge_color": "gray",
            "min_node_size": 3,
            "max_node_size": 15,
            "edge_width": 0.5,
            "width": '100vw',
            "height": '100vh',
        }
        options = {**default_options, **options}
    
        # Compute UMAP projections for node embeddings
        node_embeddings = np.array([node.embedding for node in knowledge_graph.nodes])
        umap_model = umap.UMAP(n_neighbors=30, min_dist=0.05, metric="cosine", n_components=2)
        umap_embeddings = umap_model.fit_transform(node_embeddings)
        umap_embeddings_normalized = (umap_embeddings - umap_embeddings.min(axis=0)) / (umap_embeddings.max(axis=0) - umap_embeddings.min(axis=0))
    
        # Calculate node degrees
        node_degrees = {node.id: 0 for node in knowledge_graph.nodes}
        for rel in knowledge_graph.rels:
            node_degrees[rel.source] += 1
            node_degrees[rel.target] += 1
    
        # Normalize node sizes
        min_degree = min(node_degrees.values())
        max_degree = max(node_degrees.values())
        degree_range = max_degree - min_degree
        normalize_size = lambda degree: options["min_node_size"] + (options["max_node_size"] - options["min_node_size"]) * (degree - min_degree) / degree_range if degree_range > 0 else options["min_node_size"]
    
        nodes = [
            {
                "id": node.id,
                "label": node.id,
                "description": getattr(node, "description", None),
                "type": getattr(node, "type", None),
                "source": ', '.join(getattr(node, "source", None)),
                "conversation_title": ', '.join(getattr(node, "conversation_title", None)),
                "authors": ', '.join(getattr(node, "authors", None)),
                "x": float(umap_embeddings_normalized[i, 0]),
                "y": float(umap_embeddings_normalized[i, 1]),
                "size": normalize_size(node_degrees[node.id])
            }
            for i, node in enumerate(knowledge_graph.nodes)
        ]
        links = [
            {"source": rel.source, "target": rel.target, "type": rel.type}
            for rel in knowledge_graph.rels
        ]
    
        graph_data = json.dumps({"nodes": nodes, "links": links})
    
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Interactive Knowledge Graph</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    font-family: Arial, sans-serif;
                }}
                #graph {{
                    width: 100%;
                    height: 100%;
                    background-color: #f0f0f0;
                }}
                .node circle {{
                    cursor: pointer;
                }}
                .link {{
                    stroke: #999;
                    stroke-opacity: 0.6;
                    cursor: pointer;
                }}
                .node text {{
                    font-size: 10px;
                    fill: #333;
                    pointer-events: none;
                }}
                #info-box {{
                    position: absolute;
                    background: white;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 10px;
                    display: none;
                    max-width: 300px;
                    z-index: 1000;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                #controls {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    z-index: 1000;
                }}
                #legend {{
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: white;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 10px;
                    z-index: 1000;
                }}
                .legend-item {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 5px;
                }}
                .legend-color {{
                    width: 20px;
                    height: 20px;
                    margin-right: 5px;
                    border-radius: 50%;
                }}
            </style>
        </head>
        <body>
            <div id="controls">
                <button id="toggleForce">Toggle Force</button>
                <button id="resetView">Reset View</button>
            </div>
            <div id="graph"></div>
            <div id="info-box"></div>
            <div id="legend"></div>
            <script>
                const graphData = {graph_data};
                console.log('Graph data:', graphData);
    
                const width = window.innerWidth;
                const height = window.innerHeight;
    
                const svg = d3.select("#graph")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
    
                const g = svg.append("g");
    
                const simulation = d3.forceSimulation(graphData.nodes)
                    .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(30))
                    .force("charge", d3.forceManyBody().strength(-30))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("x", d3.forceX(d => d.x * width).strength(0.6))
                    .force("y", d3.forceY(d => d.y * height).strength(0.6));
    
                const zoom = d3.zoom()
                    .scaleExtent([0.1, 4])
                    .on("zoom", (event) => {{
                        g.attr("transform", event.transform);
                    }});
    
                svg.call(zoom);
    
                const initialScale = 0.7;
                const initialTranslate = [width / 2, height / 2];
                svg.call(zoom.transform, d3.zoomIdentity.translate(initialTranslate[0], initialTranslate[1]).scale(initialScale));
    
                const link = g.selectAll(".link")
                    .data(graphData.links)
                    .enter().append("line")
                    .attr("class", "link")
                    .attr("stroke-width", {options['edge_width']})
                    .attr("stroke", "{options['edge_color']}")
                    .on("mouseover", showLinkInfo)
                    .on("mouseout", hideInfoBox);
    
                const node = g.selectAll(".node")
                    .data(graphData.nodes)
                    .enter().append("g")
                    .attr("class", "node")
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));
    
                const typeColor = d3.scaleOrdinal(d3.schemeCategory10);
    
                node.append("circle")
                    .attr("r", d => d.size)  // Use the calculated size
                    .style("fill", d => typeColor(d.type))
                    .on("click", showNodeInfo)
                    .on("mouseover", highlightConnections)
                    .on("mouseout", resetHighlight);
    
                node.append("text")
                    .attr("dx", 12)
                    .attr("dy", ".35em")
                    .text(d => d.label)
                    .style("fill-opacity", 0)
                    .style("font-size", "8px");
    
                simulation.on("tick", () => {{
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
    
                    node
                        .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
                }});
    
                function dragstarted(event, d) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}
    
                function dragged(event, d) {{
                    d.fx = event.x;
                    d.fy = event.y;
                }}
    
                function dragended(event, d) {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}
    
                function showNodeInfo(event, d) {{
                    showInfoBox(event, `
                        <h3>${{d.label}}</h3>
                        <p><strong>Type:</strong> ${{d.type || 'Not specified'}}</p>
                        <p><strong>Description:</strong> ${{d.description || 'No description available.'}}</p>
                        <p><strong>Thread ID:</strong> ${{d.source || 'Not specified'}}</p>
                        <p><strong>Conversation Title:</strong> ${{d.conversation_title || 'Not specified'}}</p>
                        <p><strong>Authors:</strong> ${{d.authors || 'Not specified'}}</p>
                        <p><strong>Cluster:</strong> ${{d.cluster}}</p>
                    `);
                }}
    
                function showLinkInfo(event, d) {{
                    showInfoBox(event, `
                        <h3>Relationship</h3>
                        <p><strong>Type:</strong> ${{d.type || 'Not specified'}}</p>
                        <p><strong>Source:</strong> ${{d.source.label || d.source.id}}</p>
                        <p><strong>Target:</strong> ${{d.target.label || d.target.id}}</p>
                    `);
                }}
    
                function showInfoBox(event, content) {{
                    const infoBox = d3.select("#info-box");
                    infoBox.html(content)
                        .style("display", "block")
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }}
    
                function hideInfoBox() {{
                    d3.select("#info-box").style("display", "none");
                }}
    
                function highlightConnections(event, d) {{
                    const connectedNodeIds = new Set(graphData.links
                        .filter(l => l.source.id === d.id || l.target.id === d.id)
                        .flatMap(l => [l.source.id, l.target.id]));
    
                    node.style("opacity", n => connectedNodeIds.has(n.id) ? 1 : 0.1);
                    link.style("opacity", l => l.source.id === d.id || l.target.id === d.id ? 1 : 0.1);
    
                    d3.select(this.parentNode).select("text")
                        .style("fill-opacity", 1);
                }}
    
                function resetHighlight() {{
                    node.style("opacity", 1);
                    link.style("opacity", 1);
                    node.select("text").style("fill-opacity", 0);
                }}
    
                let forceEnabled = true;
                d3.select("#toggleForce").on("click", () => {{
                    forceEnabled = !forceEnabled;
                    if (forceEnabled) {{
                        simulation.alpha(1).restart();
                    }} else {{
                        simulation.stop();
                    }}
                }});
    
                function centerGraph() {{
                    const bounds = g.node().getBBox();
                    const dx = bounds.width;
                    const dy = bounds.height;
                    const x = bounds.x + dx / 2;
                    const y = bounds.y + dy / 2;
    
                    const scale = 0.8 / Math.max(dx / width, dy / height);
                    const translate = [width / 2 - scale * x, height / 2 - scale * y];
    
                    svg.transition().duration(750).call(
                        zoom.transform,
                        d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
                    );
                }}
    
                d3.select("#resetView").on("click", centerGraph);
    
                // Initial centering
                setTimeout(centerGraph, 100);
    
                document.addEventListener('click', function(event) {{
                    const infoBox = document.getElementById('info-box');
                    const clickedInsideInfoBox = infoBox.contains(event.target);
                    const clickedOnNodeOrLink = event.target.tagName === 'circle' || event.target.tagName === 'line';
                    
                    if (!clickedInsideInfoBox && !clickedOnNodeOrLink) {{
                        infoBox.style.display = 'none';
                    }}
                }});
    
                console.log('Visualization setup complete');
            </script>
        </body>
        </html>
        """
    
        # Write to file and return
        with open("./knowledge_graph.html", "w") as f:
            f.write(html_content)
        return html_content

    #####################################
    ### SAVE / READ NETWORK FROM FILE ###
    #####################################
    def save_network_to_pkl(self,
                            knowledge_graph,
                            network_name=None):
        aggregate_graph_json = json.dumps(knowledge_graph.dict(), ensure_ascii=False, indent=4)

        ### CREATE DIRECTORY IF NECESSARY
        create_directory(self.graph_directory)
        
        ### SAVE TO GRAPH DIRECTORY
        if not network_name:
            with open(os.path.join(self.graph_directory, 'knowledge_graph.pkl'), 'wb') as f:
                pickle.dump(aggregate_graph_json, f)
            print(f"==> Network pickle saved to knowledge_graph.pkl")
        else:
            with open(os.path.join(self.graph_directory, f"{network_name}.pkl"), 'wb') as f:
                pickle.dump(aggregate_graph_json, f)
            print(f"==> Network pickle saved to {network_name}.pkl")

    def read_network_from_pickle(self,
                                 network_name=None):
        if not network_name:
            pkl_name = "knowledge_graph.pkl"
        else:
            pkl_name = f"{network_name}.pkl"
        pkl_path = os.path.join(self.graph_directory, pkl_name)
        with open(pkl_path, 'rb') as f:
            aggregate_graph_json = pickle.load(f)
        aggregate_graph_dict = json.loads(aggregate_graph_json)
        knowledge_graph = KnowledgeGraph(**aggregate_graph_dict)
        return knowledge_graph

