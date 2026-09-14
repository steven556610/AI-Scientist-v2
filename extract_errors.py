import json
import sys

def main():
    journal_path = sys.argv[1]
    with open(journal_path, "r") as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
        
    for i, v in enumerate(nodes):
        if isinstance(v, dict) and v.get("is_buggy"):
            print(f"Node {i} (id: {v.get('node_id')}):")
            
            exc_type = v.get("exc_type")
            exc_info = v.get("exc_info")
            if exc_type:
                print(f"Exception Type: {exc_type}")
                if exc_info:
                    print(f"Exception Info: {exc_info}")
            else:
                term_out = v.get("_term_out")
                if term_out:
                    print(f"Terminal Out (last 500 chars): {term_out[-500:]}")
            print("-" * 50)

if __name__ == "__main__":
    main()
