import random

class SILBuffer:
    """
    Self-Imitation Learning (SIL) Expert Buffer.
    Maintains a priority queue of the best historical trajectories (lowest makespan).
    """
    def __init__(self, capacity=10):
        self.capacity = capacity
        # List of dictionaries: {'makespan': float, 'transitions': list}
        self.trajectories = []
        
    def add_episode(self, makespan, memory_obj, gamma):
        """
        Adds an episode to the SIL buffer if it's better than the worst in the buffer,
        or if the buffer is not yet full.
        """
        # Exclude deadlocks (makespan >= 9999 or equivalent)
        if makespan > 2400.0: # Arbitrary high threshold to discard deadlocks
            return False

        # If full and makespan is worse than our worst stored, reject immediately
        if len(self.trajectories) >= self.capacity:
            if makespan >= self.trajectories[-1]['makespan']:
                return False
                
        # Calculate Monte-Carlo (MC) returns for this episode
        returns = []
        discounted_reward = 0
        for reward in reversed(memory_obj.rewards):
            discounted_reward = reward + gamma * discounted_reward
            returns.insert(0, discounted_reward)
            
        # Extract transitions
        transitions = []
        for i in range(len(memory_obj.states)):
            transitions.append({
                'state_snap': memory_obj.states[i],
                'action': memory_obj.actions[i],
                'mask': memory_obj.masks[i] if i < len(memory_obj.masks) else None,
                'return': returns[i]
            })
            
        self.trajectories.append({
            'makespan': makespan,
            'transitions': transitions
        })
        
        # Sort trajectories by makespan ascending (lower is better)
        self.trajectories.sort(key=lambda x: x['makespan'])
        
        # Enforce capacity
        if len(self.trajectories) > self.capacity:
            self.trajectories.pop() # Discard the highest makespan (the worst one)
            
        return True
        
    def sample_batch(self, batch_size):
        """
        Randomly samples a batch of transitions from the expert trajectories.
        For simplicity, we use uniform sampling over the flattened transitions.
        """
        all_trans = []
        for traj in self.trajectories:
            all_trans.extend(traj['transitions'])
            
        if len(all_trans) == 0:
            return []
            
        batch_size = min(batch_size, len(all_trans))
        return random.sample(all_trans, batch_size)
    
    def __len__(self):
        return sum(len(traj['transitions']) for traj in self.trajectories)

