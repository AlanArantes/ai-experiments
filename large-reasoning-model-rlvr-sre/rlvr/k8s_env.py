import gymnasium as gym
from gymnasium import spaces
import subprocess
import requests
import json
import time

class KubernetesEnv(gym.Env):
    """
    Gymnasium environment that exposes an ephemeral Kubernetes cluster
    as a reinforcement learning environment for SRE reasoning agents.
    """
    def __init__(self, kubeconfig_path: str, scenario: dict):
        super().__init__()
        self.kubeconfig = kubeconfig_path
        self.scenario = scenario
        self.step_count = 0
        self.max_steps = 10
        
        # Tools: [execute_kubectl, read_logs, curl_endpoint, rollback_deploy, edit_spec]
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(5),
            "command": spaces.Text(max_length=512)
        })
        self.observation_space = spaces.Text(max_length=4096)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # 1. Reset cluster state to clean base
        if self.kubeconfig:
            subprocess.run(["kubectl", "--kubeconfig", self.kubeconfig, "delete", "ns", "sandbox-app", "--ignore-not-found"], capture_output=True)
            subprocess.run(["kubectl", "--kubeconfig", self.kubeconfig, "create", "ns", "sandbox-app"], capture_output=True)
            
            # 2. Inject fault scenario
            if "setup_script" in self.scenario:
                subprocess.run(["bash", self.scenario["setup_script"], self.kubeconfig], capture_output=True)
        
        # 3. Return initial observation (active alerts / symptoms)
        initial_obs = f"ALERT: ServiceDegraded in namespace sandbox-app. Target: {self.scenario.get('description', '')}"
        return initial_obs, {}

    def step(self, action_dict: dict):
        self.step_count += 1
        cmd = action_dict.get("command", "")
        
        # Safety Gate: Trap catastrophic cluster destruction
        forbidden = self.scenario.get("forbidden_actions", ["kube-system", "--all", "delete node"])
        if any(bad_cmd in cmd for bad_cmd in forbidden):
            return "Action blocked by safety policy.", -2.0, True, False, {"error": "blast_radius_violation"}

        # Execute agent tool
        try:
            res = subprocess.run(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                text=True
            )
            observation = res.stdout if res.returncode == 0 else f"ERROR: {res.stderr}"
        except Exception as e:
            observation = f"Execution failed: {str(e)}"

        # Verify Environment Health (The Verifiable Reward)
        is_healthy = self._verify_service_slo()
        done = is_healthy or (self.step_count >= self.max_steps)
        
        reward = 0.0
        if is_healthy:
            reward = 1.0 - (self.step_count * 0.05)  # Reward resolution + speed bonus
        elif done and not is_healthy:
            reward = -0.5  # Max steps exhausted without fix

        return observation[:4096], reward, done, False, {}

    def _verify_service_slo(self) -> bool:
        probe_url = self.scenario.get("verification_probe")
        if not probe_url:
            return False
        try:
            # Poll probe 3 times to confirm stability, not a transient fluke
            for _ in range(3):
                r = requests.get(probe_url, timeout=2)
                if r.status_code != 200:
                    return False
                time.sleep(1)
            return True
        except requests.RequestException:
            return False
