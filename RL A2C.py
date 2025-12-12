# %% [markdown]
# # A2C Intrinsic Curiosity Agent Implementation for Atari environments

# %%
from ale_py.vector_env import AtariVectorEnv
import torch
import numpy as np
import time

# %%
num_actions=18
# Create the encoder
class state_encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Flatten = torch.nn.Flatten()
        self.ELU = torch.nn.ELU()

        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x=self.ELU(x)
        x = self.conv2(x)
        x=self.ELU(x)
        x = self.conv3(x)
        x=self.ELU(x)
        x = self.conv4(x)
        x = self.Flatten(x)
        return x
    
# Create the Inverse dynamics module
class InvDyn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = state_encoder()  # This is the state feature encoder
        self.Flatten = torch.nn.Flatten()
        self.ELU = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=1)

        self.l1 = torch.nn.Linear(in_features=72, out_features=256) # 3528 should be replaced by 2*(output shape of encoder)
        self.l2 = torch.nn.Linear(in_features=256, out_features=num_actions)

    def forward(self, s_t, s_t1):
        # Encode the two states
        e_t = self.encoder(s_t)
        e_t1 = self.encoder(s_t1)
        
        # Concatenate the layers
        x = torch.cat((e_t, e_t1), dim=1)

        # Forward propagate the encoded features through the rest of the network
        x = self.l1(x)
        x = self.ELU(x)
        x = self.l2(x)
        return x
    
encoding_output_size = 36
class Forward(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ELU = torch.nn.ELU()

        self.l1 = torch.nn.Linear(in_features=encoding_output_size+num_actions, out_features=256)
        self.l2 = torch.nn.Linear(in_features=256, out_features=encoding_output_size)

    def forward(self, s_t_encoded, a_t):
        """
        This will output an estimated encoding of s_{t+1}.

        Args:
            s_t_encoded :   feature encoding of the state s_t of time t. This should be a 2D tensor.
            a_t         :   The action taken at time t as an integer. This should be a 1D tensor.
        """
        # One-hot encode the action to be input into the network calculation
        a_t_one_hot = torch.zeros(size=(a_t.shape[0], num_actions), dtype=torch.float)
        a_t_one_hot[range(a_t.shape[0]), a_t] = 1.0

        # Calculate the output
        x = torch.cat((s_t_encoded, a_t_one_hot), dim=1)
        x = self.l1(x)
        x = self.ELU(x)
        x = self.l2(x)
        return x
    
# Create a class for complete intrinsic curiosity modules
class icm_module():
    def __init__(self):
        self.inverse_dynamics_model = InvDyn()
        self.forward_model = Forward()
    
# Create the actor class
class policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Flatten = torch.nn.Flatten()
        self.ELU = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim = 1)

        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.linear1 = torch.nn.Linear(in_features=36, out_features=num_actions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ELU(x)
        x = self.conv2(x)
        x = self.ELU(x)
        x = self.conv3(x)
        x = self.ELU(x)
        x = self.conv4(x)
        x = self.ELU(x)
        x = self.Flatten(x)
        x = self.linear1(x)
        return x
    
# Create the value class
class value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Flatten = torch.nn.Flatten()
        self.ELU = torch.nn.ELU()

        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.linear1 = torch.nn.Linear(in_features=36, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ELU(x)
        x = self.conv2(x)
        x = self.ELU(x)
        x = self.conv3(x)
        x = self.ELU(x)
        x = self.conv4(x)
        x = self.ELU(x)
        x = self.Flatten(x)
        x = self.linear1(x)
        return x

# Select the GPU to train the networks if available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define the networks (and send them to whichever device you are using)
icm = icm_module()
icm.forward_model = icm.forward_model.to(device)
icm.inverse_dynamics_model = icm.inverse_dynamics_model.to(device)
actor = policy().to(device)
critic = value().to(device)

# Select the optimizer
optimizer = torch.optim.SGD([
    {'params':icm.forward_model.parameters(),'lr':0.001},
    {'params':icm.inverse_dynamics_model.parameters(),'lr':0.001},
    {'params':actor.parameters(),'lr':0.001},
    {'params':critic.parameters(),'lr':0.001}
    ])

# Initialize update_num
update_num = 0

# Initialize file name to store agent progress
agent_file = 'agent_checkpoint.pth'

# Load the models if they already exist
try:
    checkpoint = torch.load(agent_file, map_location=device)

    update_num = checkpoint['update number']

    icm.forward_model.load_state_dict(checkpoint['icm forward'])
    icm.inverse_dynamics_model.load_state_dict(checkpoint['icm inverse dynamics'])
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])

    # Load the optimizer in its last saved state if you want to continue training the model
    optimizer_loaded = torch.optim.SGD([
        {'params':icm.forward_model.parameters(),'lr':0.001},
        {'params':icm.inverse_dynamics_model.parameters(),'lr':0.001},
        {'params':actor.parameters(),'lr':0.001},
        {'params':critic.parameters(),'lr':0.001}
        ])  # Same as the optimizer used in training

    optimizer.load_state_dict(checkpoint['optimizer state'])

except:
    None

# %%
# Training duration variables
num_workers = 10            # Number of separate worker agents
num_steps_to_update = 5     # The number of environment interaction the agent will make in each environment
num_updates = 40000         # Total number of parameter updates 
save_interval = 100         # Number of updates until progress is saved

eta = 1         # Scalar for intrinsic rewards (must be greater than 0)
gamma = 1       # Scalar for critic targets
beta = 0.2      # Scalar to balance the effects of the forward and inverse dynamic loss
lambd = 0.1




# Create a vector environment with 4 parallel instances of Breakout
envs = AtariVectorEnv(
    game="montezuma_revenge",  # The ROM id not name, i.e., camel case compared to `gymnasium.make` name versions
    num_envs=num_workers,
    grayscale=True
)

# Reset all environments
observations, info = envs.reset()

observations = observations/255.0

# Initialize all models
start_time = time.time()
for i in range(update_num, update_num + num_updates):
    # Initialize the arrays to store transitions
    observation_array = np.zeros(shape = (0, *observations.shape), dtype = float)
    ex_reward_array = np.zeros(shape = (0, num_workers), dtype = float) # Extrinsic rewards
    terminated_array = np.zeros(shape = (0, num_workers), dtype = float)
    truncated_array = np.zeros(shape = (0, num_workers), dtype = float)
    action_array = np.zeros(shape = (0, num_workers), dtype = float)
    in_reward_array = np.zeros(shape = (0, num_workers), dtype = float) # Intrinsic rewards

    # Add the first observation of this batch of transitions to the observation_array
    observation_array = np.vstack((observation_array, observations.reshape(1, *observations.shape)))

    # Take some amount of steps in the environment
    for j in range(num_steps_to_update):
        ### START OF POLICY
        with torch.no_grad():
            action_probabilities = actor.softmax(actor(torch.tensor(observations.reshape(observations.shape[0], 1, *observations.shape[1:]), dtype=torch.float))).numpy()
        actions = np.random.choice(range(num_actions), size=(num_workers,))
        ### END OF POLICY

        # Interact with the environment to collect transitions
        observations, ex_rewards, terminations, truncations, infos = envs.step(actions)
        observations = observations/255.0

        # Save the transitions
        action_array = np.vstack((action_array, actions))
        ex_reward_array = np.vstack((ex_reward_array, ex_rewards))
        terminated_array = np.vstack((terminated_array, terminations))
        truncated_array = np.vstack((truncated_array, truncations))
        observation_array = np.vstack((observation_array, observations.reshape(1, *observations.shape)))
    
    # Initialize loss to zero
    loss = 0

    # Compute and save ICM intrinsic rewards
    for k in range(num_workers):
        # Get the transitions for this worker
        o = torch.tensor(observation_array[:, k:k+1], dtype=torch.float).to(device)
        a = torch.tensor(action_array[:, k:k+1], dtype=torch.long).to(device)
        ex_rewards = torch.tensor(ex_reward_array[:, k:k+1], dtype=torch.float).to(device)
        term = torch.tensor(terminated_array[:, k:k+1], dtype=torch.float).to(device)
        trun = torch.tensor(truncated_array[:, k:k+1], dtype=torch.float).to(device)

        # Get the true state encodings
        true_encodings = icm.inverse_dynamics_model.encoder(o)

        # Calculate the intrinsic rewards
        pred_encodings = icm.forward_model(true_encodings[:num_steps_to_update], a)
        in_rewards = (eta/2)*torch.reshape(torch.linalg.norm(true_encodings[1:] - pred_encodings, dim=1), shape=(num_steps_to_update, 1))

        ### CALCULATE THE LOSSES TO TRAIN THE NETWORK
        # ICM:
        # Compute the ICM losses
        action_predictions = icm.inverse_dynamics_model(o[:num_steps_to_update], o[1:]).detach()
        loss_f = torch.nn.MSELoss()(pred_encodings, true_encodings[1:])                                         # forward loss
        loss_i = torch.nn.CrossEntropyLoss()(action_predictions, torch.reshape(a, shape = (a.shape[0],)))  # inverse dynamics loss

        # Critic:
        # Calculate the critic targets and the critic loss
        critic_targets = (ex_rewards + in_rewards) + gamma*critic(o[1:])*(1-term)
        loss_critic = torch.nn.MSELoss()(critic(o[:num_steps_to_update]), critic_targets)

        # Actor:
        # Calculate the advantage and the actor loss
        advantage = torch.reshape(critic_targets - critic(o[:num_steps_to_update]), shape=(critic_targets.shape[0],))
        loss_actor = torch.nn.CrossEntropyLoss(reduce=None)(actor(o[:num_steps_to_update]), torch.reshape(a, shape=(a.shape[0],)))*advantage
        loss_actor = torch.sum(loss_actor)

        # Store the losses
        loss += (loss_f*beta + loss_i*(1-beta) + (loss_critic + loss_actor)*lambd)/num_workers

    # Clear any previously calculated gradients, back-propagate, and update the parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  

    # Save the models at some milestones
    if (update_num + i)%save_interval == 0:
        torch.save({
                'optimizer state' : optimizer.state_dict(),
                'icm forward' : icm.forward_model.state_dict(),
                'icm inverse dynamics' : icm.inverse_dynamics_model.state_dict(),
                'actor' : actor.state_dict(),
                'critic' : critic.state_dict(),
                'loss' : loss,
                'update number' : i
            }, agent_file)
        print(f'Update {i}/{update_num + num_updates}\nloss: {loss}\nTime taken to complete the last {save_interval} updates: {time.time() - start_time}\n-----------------')
        start_time = time.time()
        
    

# Close the environment when done
envs.close()
print('DONE!')