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
    repo = git.Repo(project_root)
    
    try: 
        # Add all changes to the staging area
        repo.git.add(all=True)
        
        # Stash changes
        repo.git.stash("save")
        
        # Get the experiments branch
        experiments_branch = repo.branches["experiments"]

        # Checkout the experiments branch
        repo.git.checkout("experiments")
        
        # Pop the stash
        repo.git.stash("pop")

        if repo.is_dirty(untracked_files=True): 
            # Add all changes to the staging area
            repo.git.add(all=True)

            # Commit the changes to the experiments branch
            repo.git.commit(message="Auto-commit to experiments branch.")

            # Push the changes to the remote repository
            repo.remote().push(experiments_branch)
            
        # Get the commit hash values
        commit_hash = repo.head.commit.hexsha
        print(f"Commit hash: {commit_hash}")
    except Exception as e:
        print(f"Error: {e}")
        print("An error occurred while committing to the experiments branch.")
    
    # Go back to main branch
    experiments_branch = repo.branches["main"]

    # Checkout the experiments branch
    repo.git.checkout("main")
    
    return commit_hash
            
    
