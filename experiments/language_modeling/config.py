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
    
    try:         
        if repo.is_dirty(untracked_files=True): 
            print("Committing untracked files on `main`...")
            # Add all changes to the staging area
            repo.git.add(all=True)

            # Commit the changes to the experiments branch
            repo.git.commit(message="Auto-commit to `main` branch.")

            # Push the changes to the remote repository
            repo.remote().push(main_branch)
        
        # Accept incoming changes on the new branch
        repo.git.merge("--strategy=ours", "experiments")
        print("merge done")

        # Checkout the experiments branch
        repo.git.checkout("experiments")
        
        # Apply the stash
        repo.git.stash("apply")

        # Check if there are any merge conflicts
        if repo.git.unmerged_files():
            print("Merge conflicts detected. Aborting commit.")
            repo.git.stash("drop")

        if repo.is_dirty(untracked_files=True): 
            print("Committing untracked files...")
            # Add all changes to the staging area
            repo.git.add(all=True)

            # Commit the changes to the experiments branch
            repo.git.commit(message="Auto-commit to `experiments` branch.")

            # Push the changes to the remote repository
            repo.remote().push(experiments_branch)
        else:
            print("No changes to commit.")
            
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
    
