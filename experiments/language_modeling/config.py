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
    main_branch = repo.branches["main"]
    experiments_branch = repo.branches["experiments"]
    
    print(f"Committing current codebase under {project_root} to the `experiments` branch...")
    print("pimmel")
    try:       
        repo.git.checkout("main")  
        if repo.is_dirty(untracked_files=True): 
            print("Committing untracked files...")
            # Add all changes to the staging area
            repo.git.add(all=True)

            # Commit the changes to the experiments branch
            repo.git.commit(message="Auto-commit to `main` branch.")

            # Push the changes to the remote repository
            repo.remote().push(main_branch)
        
        # Accept incoming changes on the new branch
        repo.git.merge("experiments", X="ours")

        # Checkout the experiments branch
        repo.git.checkout("experiments")

        if repo.is_dirty(untracked_files=True): 
            print("Committing untracked files...")
            # Add all changes to the staging area
            repo.git.add(all=True)

            # Commit the changes to the experiments branch
            repo.git.commit(message="Auto-commit to `experiments` branch.")

            # Push the changes to the remote repository
            
        else:
            print("No changes to commit.")
            
        repo.remote().push(experiments_branch)
            
        # Get the commit hash values
        commit_hash = repo.head.commit.hexsha
        print(f"Commit hash: {commit_hash}")
    except Exception as e:
        print(f"Error: {e}")
        print("An error occurred while committing to the experiments branch.")
    
    # Checkout the  branch
    repo.git.checkout("main")
        
    return commit_hash
    
