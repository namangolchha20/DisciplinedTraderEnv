import uvicorn

from openenv.core.env_server import create_app
from env.models import Action, Observation
from env.environment import DisciplinedTraderEnv

app = create_app(DisciplinedTraderEnv, Action, Observation, env_name="disciplined_trader_env")

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()