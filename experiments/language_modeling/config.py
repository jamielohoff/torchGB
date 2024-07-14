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
    # Open the repository
    repo = git.Repo(project_root)    
    
    # Get the experiments branch
    experiments_branch = repo.branches["experiments"]
    
    print(f"Committing current codebase under {project_root} to the `experiments` branch...")
    print("test")
    try:         
        # Stash changes
        repo.git.stash("save")
        
        # Add local changes
        repo.git.add(all=True)
        
        # Accept incoming changes on the new branch
        repo.git.merge("--strategy=ours", "experiments")
        print("merge done")

        # Checkout the experiments branch
        repo.git.checkout("experiments")
        
        # Pop the stash
        repo.git.stash("apply")

        if repo.is_dirty(untracked_files=True): 
            print("Committing untracked files...")
            # Add all changes to the staging area
            repo.git.add(all=True)

            # Commit the changes to the experiments branch
            repo.git.commit(message="Auto-commit to `experiments` branch.")

            # Push the changes to the remote repository
            repo.remote().push(experiments_branch)
            
        # Get the commit hash values
        commit_hash = repo.head.commit.hexsha
        print(f"Commit hash: {commit_hash}")
    except Exception as e:
        print(f"Error: {e}")
        print("An error occurred while committing to the experiments branch.")
    
    # Checkout the  branch
    repo.git.checkout("main")
    
    # Reapply the stashed stuff
    repo.git.stash("pop")
    
    return commit_hash
    
