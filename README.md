# Multi-LLM Agent Meta-Benchmarking Method
A business-process inspired multiagent meta benchmarking method for generating AI Safety benchmarks with applications in scalable content moderation and compliance. 

Given only a topic (e.g.: home repair), our method serves as an API for a multi-agent business-centered compliance and safety check. 

Perhaps the most critical and underlooked barrier to widespread adoption of AI safety paradigms outside of research communities is compatibility with business-centric requirements, especially for tech-laggard industries looking to adopt AI (e.g. home repair). Core to the common business requirements, non-technical stakeholders must understand the risks associated with AI technology, with technology release only occurring after satisfying a high bar for ethical, liability, and financial risk reasons. The bar will depend on a variety of factors, including the organization, industry, location, and ever changing user behaviors. Our work embeds industry standard risk mitigation processes into a new, robust multi-agent framework flexible enough to meet these needs, forming the first business-viable benchmark. 

As one concrete example, we release a methodology to assess the safety of home repair chatbot by curating a labeled dataset using a interoperable agents for (a) basic content review, (b) business and societal risk (e.g., compliance, customer satisfaction), and (c) legal review. We also provide a example dataset. The approach is easily extensible more broadly to other topics and can be iterated upon via a simple python package, which we hope will lead to a comprehensive benchmarking paradigm. Ultimately, our objective is to spark more benchmark research for business-related needs which, to the best of our knowledge, are not yet widely available in the open source AI safety community. 


## Instructions 

## Run Instructions

To install dependencies and run the program, use the following commands:

```sh
pip install -r requirements.txt
python3 meta_benchmarking_process.py <"OpenAI Key"> <"topic"> <"location of generated_text_csv_to_benchmark_and_score">
```
For example: 
python3 meta_benchmarking_process.py "pretend_open_AI_Key" "home repair" "sample.csv"

The generated guidelines are printed to the screen and the raw labels are saved in a DataFrame. Rows represent `runs' while columns represent prompt number to be labeled. 

## Results 

|Baseline|   Classified Output+ % Safe | Classified Output+ % Human Review |  Classified Output+ % Unsafe | Across Runs % Safe | Across Runs % Human Review | Across Runs % Unsafe |
|------------|---------------|----------------------|---------------|---------------------------|------------------------------|---------------------------|
| **Simple** | 0.99          | 0.01                 | 0            | 0.99                      | 0.003                        | 0                         |
| **Complex**| 0.72          | 0.26                 | 0.02         | 0.82                      | 0.17                         | 0.01                      |



+ The **classified output** metric takes each of the runs and “maxes” over any unsafe or human review outputs. This is the benefit of running a Monte Carlo simulation via multiple experimental runs, as required by the domain -- because if, in any run, the prompt were ultimately labeled ‘unsafe’ or ‘human review’, that is the label it will retain. As discussed in the future works section, there could be some nuance based on the domain on this label aggregation scheme. In our case, our ChatGPT agents still had several prompts that needed human review on the simple baseline, which matched the sensitivity required for this application, and overall quite a ways to go towards progress on the complex baseline.


* The **across runs** metric simply considers all the labels equally weighted regardless of what prompt they belong to. In this metric, the simple baseline generally reports safe labels over half the time. The difference between the overall output and across metrics support what we empirically observe: labels tend to be fairly consistent per prompt, with no prompts returning both safe and unsafe labels across different runs. This sanity check supports the integrity of our evaluation process. 
