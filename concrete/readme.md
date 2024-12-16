The project was completed in two main stages. The first stage involved exploratory data
engineering and analysis, as well as testing various Machine Learning (ML) models to enhance
performance metrics. In the second stage, a graphical user interface (GUI) was developed, and
the best-performing ML model was refined to accept inputs from engineers and output
estimated compressive strengths. Additionally, the GUI could generate graphs displaying the
progression of strength over time.
A key insight gained during the project was that not only the abundance of data but also the
variety of data is critical in developing robust ML models for real-world engineering
applications. While the GUI and ML model can predict compressive strength with acceptable
regression metrics, it is limited to only eight variables related to concrete mixtures. In real
engineering practice, other factors such as laboratory temperature, curing methods, specimen
shape, and other aspects may significantly influence strength.
To make concrete strength estimation using Machine Learning practical in civil engineering, it
would be beneficial to gather large, diverse datasets with significantly more features. Training
such datasets on Deep Neural Networks with varying architectures could lead to substantial
improvements in regression metrics.
With enhanced datasets, it might also be possible to implement Abram’s law as a baseline for
comparing ML model predictions. Furthermore, it was notable that regression models such as
Random Forest and Gradient Boost performed well. Testing these models on larger datasets
would provide valuable insights into how their performance metrics compare to those of Deep
Neural Networks.
In conclusion, developing practical software for this purpose requires significantly larger
datasets with more diverse features to improve performance. Such advancements could
encourage engineering companies to adopt the solution as a beta for real-world applic
