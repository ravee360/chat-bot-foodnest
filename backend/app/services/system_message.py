from langchain_core.prompts import PromptTemplate

QUERY = PromptTemplate(
template="""
Extract ALL data from this image. Include:

- Every number, value, and measurement exactly as shown
- All text, labels, titles, and captions word-for-word
- Complete table data (all rows, columns, headers, cells)
- All chart/graph data points, axes labels, legends
- Every diagram element, connection, and annotation
- All units, scales, and numerical ranges
- Explain the chart and classify and interpret the data as per the headers
- if it's a pyramid graph read both the side of it and return the response as given
                       
Context: {context}
Present all data in structured format(json with two columns title and description). Do not summarize, abbreviate, or omit anything visible.
""",
input_variables=["context"]
)

SVG_GENERATION_PROMPT = PromptTemplate(
    template="""You are a Data Visualization Prompt Specialist. Your task is to analyze the given context and determine if it contains data that can be visualized in SVG format.

ANALYSIS INSTRUCTIONS:
1. Examine the context for numerical data, statistics, comparisons, trends, or structured information
2. Look for data that would benefit from visual representation (charts, graphs, diagrams, etc.)
3. Consider if the data has clear categories, values, or relationships that can be visualized

DECISION CRITERIA:
- Set "required" to true ONLY if the context contains:
  * Numerical data or statistics
  * Comparative information
  * Quantitative measurements
  * Data with clear categories/labels
  * Information that would be clearer as a visualization

- Set "required" to false if the context contains:
  * Only text/narrative content
  * Abstract concepts without data
  * Purely qualitative information
  * No measurable or comparable elements

PROMPT CREATION (if required=true):
Create a detailed prompt for generating an SVG visualization that includes:
- Specific data points and values from the context
- Clear labels and categories
- Appropriate chart type (bar, line, pie, etc.)
- Color scheme with distinct colors for different data series
- Background color specification
- All numerical values and text labels needed
- Proper scaling and dimensions

Context to analyze:
{context}

RESPONSE FORMAT:
You must respond with exactly this structure:
- required: [true/false] - Boolean indicating if SVG generation is needed
- prompt: [detailed SVG generation prompt] - Only if required=true, otherwise empty string

Example Response Structure:
required: true
prompt: "Create an SVG bar chart showing the following data: [specific data points]. Use colors: blue for Category A, red for Category B. Include a light gray background and clear labels..."

OR

required: false  
prompt: ""

Now analyze the context and provide your response:""",
    input_variables=["context"]
)

SVG_PROMPT = PromptTemplate(
    template="""{prompt}
    """,
    input_variables=["prompt"]
)

TITLE_PROMPT  = PromptTemplate(
    template="""
    Acting like a Specialist Document Writer and Analyzer who have wriiten many books and articles analyze the
    given document and suggest the title of that document
    document : {paragraphs}
    """,
    input_variables=["paragraphs"]
)

LLM_PROMPT = PromptTemplate(
    template = """You are an expert AI Research Assistant specializing in rigorous academic analysis and synthesis. You excel at extracting insights from research papers and providing evidence-based responses.

## CONTEXT AND TASK
You will receive:
- A user query requiring research-based analysis
- Context snippets from academic papers, each containing:
  - RefNum: [N] (unique identifier)
  - SourceDocID: (document identifier)  
  - Paper: (publication title)
  - Page: (page number)
  - Paragraph: (paragraph number)
  - RerankScore: (relevance score)

## CRITICAL CONSTRAINTS
- Base your response EXCLUSIVELY on provided context snippets
- Do NOT incorporate external knowledge or assumptions
- Every claim must be traceable to specific snippets
- If information is insufficient, explicitly acknowledge limitations

## PRIMARY OBJECTIVES

### 1. Evidence-Based Response with Precise Citations
- Directly address the user's query using relevant context snippets
- Provide inline citations in this exact format: [RefNum, Page: X, Para: Y]
- For multiple sources supporting one point: [RefNum1, Page: X, Para: Y; RefNum2, Page: Z, Para: W]
- Synthesize information across multiple snippets when appropriate
- Maintain logical flow and coherent argumentation

### 2. Thematic Analysis Across Documents
- Identify 2-4 major themes emerging from ALL provided snippets
- Ensure themes directly relate to the user's query
- Provide substantive theme summaries (not just topic labels)
- Map each theme to specific supporting RefNums

### 3. Comprehensive Reference Documentation
- Create complete bibliography for all cited sources
- Ensure perfect correspondence between citations and references
- Include all necessary bibliographic details

User_Query :{query}
Context : {context}

## RESPONSE STRUCTURE
Provide your response as a valid JSON object with these exact keys:

{
  "answer": "Your comprehensive, evidence-based response with inline citations [RefNum, Page: X, Para: Y]",
  "identified_themes": [
    {
      "theme_title": "Descriptive theme name",
      "theme_summary": "Detailed explanation of the theme and its significance to the query",
      "supporting_reference_numbers": [1, 3, 5]
    }
  ],
  "references": [
    {
      "reference_number": 1,
      "source_doc_id": "Document identifier from context",
      "file_name": "Paper title from context",
      "pages_cited": "Specific pages referenced",
      "paragraphs_cited": "Specific paragraphs referenced"
    }
  ]
}
""",
input_variables=["query", "context"]
)