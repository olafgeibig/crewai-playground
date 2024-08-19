## Agent Attributes[¶](https://docs.crewai.com/core-concepts/Agents/#agent-attributes "Permanent link")

|Attribute|Parameter|Description|
|---|---|---|
|**Role**|`role`|Defines the agent's function within the crew. It determines the kind of tasks the agent is best suited for.|
|**Goal**|`goal`|The individual objective that the agent aims to achieve. It guides the agent's decision-making process.|
|**Backstory**|`backstory`|Provides context to the agent's role and goal, enriching the interaction and collaboration dynamics.|
|**LLM** _(optional)_|`llm`|Represents the language model that will run the agent. It dynamically fetches the model name from the `OPENAI_MODEL_NAME` environment variable, defaulting to "gpt-4" if not specified.|
|**Tools** _(optional)_|`tools`|Set of capabilities or functions that the agent can use to perform tasks. Expected to be instances of custom classes compatible with the agent's execution environment. Tools are initialized with a default value of an empty list.|
|**Function Calling LLM** _(optional)_|`function_calling_llm`|Specifies the language model that will handle the tool calling for this agent, overriding the crew function calling LLM if passed. Default is `None`.|
|**Max Iter** _(optional)_|`max_iter`|Max Iter is the maximum number of iterations the agent can perform before being forced to give its best answer. Default is `25`.|
|**Max RPM** _(optional)_|`max_rpm`|Max RPM is the maximum number of requests per minute the agent can perform to avoid rate limits. It's optional and can be left unspecified, with a default value of `None`.|
|**Max Execution Time** _(optional)_|`max_execution_time`|Max Execution Time is the maximum execution time for an agent to execute a task. It's optional and can be left unspecified, with a default value of `None`, meaning no max execution time.|
|**Verbose** _(optional)_|`verbose`|Setting this to `True` configures the internal logger to provide detailed execution logs, aiding in debugging and monitoring. Default is `False`.|
|**Allow Delegation** _(optional)_|`allow_delegation`|Agents can delegate tasks or questions to one another, ensuring that each task is handled by the most suitable agent. Default is `True`.|
|**Step Callback** _(optional)_|`step_callback`|A function that is called after each step of the agent. This can be used to log the agent's actions or to perform other operations. It will overwrite the crew `step_callback`.|
|**Cache** _(optional)_|`cache`|Indicates if the agent should use a cache for tool usage. Default is `True`.|
|**System Template** _(optional)_|`system_template`|Specifies the system format for the agent. Default is `None`.|
|**Prompt Template** _(optional)_|`prompt_template`|Specifies the prompt format for the agent. Default is `None`.|
|**Response Template** _(optional)_|`response_template`|Specifies the response format for the agent. Default is `None`.|
|**Allow Code Execution** _(optional)_|`allow_code_execution`|Enable code execution for the agent. Default is `False`.|
|**Max Retry Limit** _(optional)_|`max_retry_limit`|Maximum number of retries for an agent to execute a task when an error occurs. Default is `2`.|