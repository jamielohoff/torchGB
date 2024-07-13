import yaml
import git


def load_config(file: str) -> dict:
    with open(file) as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    return conf


def commit_to_experiments_branch(project_root: str):
    print("Committing current codebase to the `experiments` branch...")
    # Open the repository
    print(project_root)
    repo = git.Repo(project_root)

    # Get the experiments branch
    print(repo.branches)
    experiments_branch = repo.branches["experiments"]

    # Checkout the experiments branch
    repo.git.checkout("experiments")
    
    # Add all changes to the staging area
    repo.git.add(all=True)

    # Commit the changes to the experiments branch
    repo.git.commit("Auto-commit to experiments branch.")
    print("committing")

    
    # Push the changes to the remote repository
    repo.remote().push(experiments_branch)

    # Get the commit hash values
    commit_hash = repo.head.commit.hexsha
    print(f"Commit hash: {commit_hash}")
    return commit_hash
            
    