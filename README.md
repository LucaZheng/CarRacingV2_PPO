# CarRacingV2_PPO

I decided to write this article because I recognize the immense benefits of using simple gaming environments to explore and understand how and why the PPO method works. However, I've noticed that while many researchers have experimented with this particular environment, they've often left gaps in addressing its inherent weaknesses. Therefore, in this article, I will cover the following topics, and I will also share my code and the final weights of my model:How I customized and trained my agent.
1.	The issues I have encountered and how I solved them
2.	Debugging process
The goal of this experiment is to train an agent to control a race car, enabling it to autopilot on the track and finish as quickly as possible. This experiment is particularly meaningful as it showcases the agent's ability to perform automotive driving and auto-path finding. Additionally, it requires relatively modest training resources, making it accessible for reinforcement learning researchers who want to dive deeper into their preferred model architectures.
Instead of using pre-built models like Stable-Baselines3, I chose to modify my own PPO model. While Stable-Baselines3 offers easy implementation and some adaptability, it lacks the transparency needed for in-depth customization and research. I used PyTorch for model construction, and the entire experiment was conducted on Google Colab with an A100 GPU.

**Game Condition (from Gymnasium Documentation)**
1.	Environment: A 2D car racing game by OpenAI Gymnasium. The observation space has 98x98 pixels with 3 color channels. 
2.	Action Space: Continuous float number from 0.0-1.0
o	Steering: -1, +1 is full right
o	Gas: if 0 no gas, +1 full power
o	Braking: if 0 no break, +1 full break
3.	Rules: The reward is -0.1 per frame and +1000/N for each track tile visited, where N is the total number of tiles. For example, finishing in 732 frames yields a reward of 1000 - 0.1*732 = 926.8 points.
   
****Approaches and Process:****

**Models: CNN + PPO**
The agent network is structured with a CNN-actor network and a CNN-critic network using PyTorch. Each CNN is composed of 2 shallow layers with ReLU activation. These CNN networks receive pre-processed image data from the environment as input and extract important features, which are then fed forward into the acting and critic MLP networks.

**Optimizer**
Both networks use Adam as their optimizer with a learning rate of 0.0003. No learning rate scheduler is used, and the results are reasonably stable.
Preprocessing the Environment and Actions
This is the most crucial step before training. The first two networks that receive signals from the environment are convolutional nets. The goal is to eliminate unnecessary information and magnify essential information from the observation inputs to maximize training efficiency.

**Observation Space:**
1.	Transform the Original Observation Image: Convert the RGB scale images to Grayscale and stack 4 frames as input.
o	Important Note: After the grayscale transformation, the color channel dimension becomes 1. Therefore, the CNN color channel input dimension becomes 4 (4 frames of 1-dimensional grayscale images).
2.	Crop the Information Panel: The environment includes an information panel that tracks the speed of the race car. This panel is cropped out before training as it does not provide necessary learnable features.
3.	Final Input Shape: After transformation, the final input shape is [batch_size, 4, 84, 84].
   
**Action Space:**
Transform the action space from continuous to discrete variables. This helps limit the possible action combination search space and reduces model training costs. In this experiment, I adapted the ‘soft action’ mapping strategy (source), where the discretized points around the continuous space are not only close to the corners. Here is the soft action mapping strategy I designed, which can be modified during the training session to meet specific expected training outcomes. This action map has a total of 21 actions, which were adapted effectively into the following training session.

**The reward customization:**
The default reward function significantly influences the agent’s behavior. Overly generous positive rewards can lead to speeding and drifting, while excessive punishments can cause the agent to focus on minimizing penalties. This often results in the agent stopping completely, causing the model loss to quickly converge to a local minimum. To address this issue, I modified the reward function by capping the positive rewards and implementing what I called a "lazy punisher" mechanism.
The lazy punisher slightly penalizes the agent for inaction or for receiving negative rewards. The punisher coefficient increases each time the agent receives a negative score. Although the lazy punisher coefficient is small, it is sufficient to signal to the agent that continuously receiving negative scores is not beneficial. Consequently, the agent learns to increase speed without becoming an overly aggressive driver.
This balanced approach helps the agent to optimize its driving strategy, promoting efficient driving behavior while avoiding the pitfalls of over-correction or excessive conservatism.

**Hyperparameters**
One of the best features of PPO is that it does not require much hyperparameter tuning. Given the scale of the environment, I used the standard hyperparameters, which worked perfectly.

**Failures Encountered and Their Solutions**

During this experiment, I encountered several problems. I documented the most significant ones as follows, and their solutions.
The ‘Lazy’ Car
One issue I faced during the initial training was what I termed the ‘Lazy car’ behavior. After several training loops, the car’s behavior converged to selecting just 1 or 2 actions repeatedly. Specifically, the car preferred the ‘braking’ action, resulting in it staying stationary throughout the game to avoid losing points.
Solution: To resolve this, I avoided using the Kaiming Normal weight initialization and instead used the default weight initialization. The exact reason why Kaiming Normal caused this issue is still unclear and requires further investigation.

**Insensitive Agent**
After 400k training steps, the training converged, and the agent chose only the ‘gas’ action to gain scores until it hit the horizon and got terminated. By ignoring the turns on the track and only speeding forward, the agent believed it would maximize the score within one game, ignoring the fact that there is an entire track to complete. 
This is a common problem I’ve seen many people have encountered. Unfortunately, I haven’t found any articles addressing the solution. I noticed that those who did not encounter this issue often used a parallel environment training strategy, which expands action exploration possibilities at certain scales. I have also noticed, at the beginning of the training session, most of the collected observation images contained actions where the agent was traveling outside the track. The agent didn’t understand the difference between being inside or outside the track, causing more confusion which impacted the model's exploration ability. Therefore, trimming the collected data is essential.
Solution: If the agent did not receive positive rewards in the previous 500 steps (indicating the car was either going backward or positioned outside the track), the game was terminated. By applying this method, the agent significantly improved its score after around 20k training steps. I am using a single environment for the training, and implementing time-out to collect only the necessary data resolved this issue perfectly. 

**The Drifting Car**
Another common issue is the car drifting off the track while attempting sharp turns. The car would often drift outside the track or even move backward.
Solution:
1.	Clipping Positive Rewards: By clipping the positive reward to 1.0, the agent overcomes the speeding problem and reduces the risk of drifting.
2.	Setting a Speed Limit: I set a speed limit (0.7 max for gas) for the car, which provides more stability when performing sharp turns, effectively improving the agent’s performance.

**Final Thoughts**

**Debugging steps I used**
In my opinion, this environment is underdeveloped but has so much potential. Without preprocessing and customization, this environment is nearly unsolvable for PPO, and the fine-tuning sessions can be particularly frustrating, as even a small mistake in your code can lead to poor results. However, difficult challenges usually lead to huge learning progress. I would like to include a quick checklist for debugging in case needed:
1.	Preprocessed Observation Batch: Are the color channels correct? Is the input size correct? Are the values normalized properly?
2.	Network Outputs: Do the outputs from the CNNs and the actor/critic networks make sense?
3.	Data Volume: Do you have enough data for the model to train effectively? Adjust the batch size based on your needs.
4.	Training Loop Logic: Is the logic correct? Are the indentations and nesting logic correct?
5.	Agent Behavior: Observe how the agent acts under certain conditions. Ask why it behaves that way and investigate the reasons.
   
**Recommendation for Improvements in the Environment**
The Car Racing environment has several limitations that make training difficult without some level of customization. Here are a few improvements that could make the training process more manageable:
1.	Time-out Function: A function that can auto-detect whether the car is outside the track, how far the car is from the track, etc.
2.	Starting Position Freedom: The ability to position the car at any point on the track as a starting point. This would allow for more effective training by focusing on tricky sections such as sharp turns.
3.	Track Randomization: Randomizing the shape of the track in each game would help the agent learn a wider range of skills and benefit from model training normalization.
Some researchers have addressed these recommendations, and you can find these customizations easily on Google. However, the most recent solutions I found were from two years ago, and some built-in functions are no longer compact.
	
**Thank notes and reference:**
Thanks notanymike for introducing the soft action mapping methods and rewards clipping methods: https://notanymike.github.io/Solving-CarRacing/

