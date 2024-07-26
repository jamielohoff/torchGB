import os
import time
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
    # Wait for your turn to access the repository
    while os.path.exists(os.path.join(project_root, ".git", "index.lock")):
        print("Waiting for the index.lock file to be released.")
        time.sleep(5)
    
    # Open the repository
    repo = git.Repo(project_root)    
    
    # Get the experiments branch
    experiments_branch = repo.branches["experiments"]
    
    try:       
        # Switch to the main branch
        repo.git.checkout("main")  
        
        if repo.is_dirty(untracked_files=True): 
            # Add all changes to the staging area
            repo.git.add(all=True)

            # Commit the changes to the experiments branch
            repo.git.commit(message="Auto-commit to `experiment` branch.")

        # Checkout the experiments branch
        repo.git.checkout("experiments")
        
        # Accept incoming changes on the new branch
        repo.git.merge("main", X="theirs")
            
        # Push the changes to the remote repository
        repo.remote().push(experiments_branch)
        
    except Exception as e:
        print(f"Error: {e}")
        print("An error occurred while committing to the experiments branch.")
        
    # Get the commit hash values
    commit_hash = repo.head.commit.hexsha
    print(f"Commit hash: {commit_hash}")
    
    # Checkout the  branch
    repo.git.checkout("main")
        
    return commit_hash
    
