# Tasks
In the crewAI framework, tasks are specific assignments completed by agents. They provide all necessary details for execution, such as a description, the agent responsible, required tools, and more, facilitating a wide range of action complexities.

Tasks within crewAI can be collaborative, requiring multiple agents to work together. This is managed through the task properties and orchestrated by the Crew's process, enhancing teamwork and efficiency.

## Task Attributes[¶](https://docs.crewai.com/core-concepts/Tasks/#task-attributes "Permanent link")

|Attribute|Parameters|Description|
|---|---|---|
|**Description**|`description`|A clear, concise statement of what the task entails.|
|**Agent**|`agent`|The agent responsible for the task, assigned either directly or by the crew's process.|
|**Expected Output**|`expected_output`|A detailed description of what the task's completion looks like.|
|**Tools** _(optional)_|`tools`|The functions or capabilities the agent can utilize to perform the task. Defaults to an empty list.|
|**Async Execution** _(optional)_|`async_execution`|If set, the task executes asynchronously, allowing progression without waiting for completion. Defaults to False.|
|**Context** _(optional)_|`context`|Specifies tasks whose outputs are used as context for this task.|
|**Config** _(optional)_|`config`|Additional configuration details for the agent executing the task, allowing further customization. Defaults to None.|
|**Output JSON** _(optional)_|`output_json`|Outputs a JSON object, requiring an OpenAI client. Only one output format can be set.|
|**Output Pydantic** _(optional)_|`output_pydantic`|Outputs a Pydantic model object, requiring an OpenAI client. Only one output format can be set.|
|**Output File** _(optional)_|`output_file`|Saves the task output to a file. If used with `Output JSON` or `Output Pydantic`, specifies how the output is saved.|
|**Output** _(optional)_|`output`|An instance of `TaskOutput`, containing the raw, JSON, and Pydantic output plus additional details.|
|**Callback** _(optional)_|`callback`|A callable that is executed with the task's output upon completion.|
|**Human Input** _(optional)_|`human_input`|Indicates if the task requires human feedback at the end, useful for tasks needing human oversight. Defaults to False.|
|**Converter Class** _(optional)_|`converter_cls`|A converter class used to export structured output. Defaults to None.|