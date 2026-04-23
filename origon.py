import argparse
import requests
import sys

def main():
    parser = argparse.ArgumentParser(description="Origon AI CLI Client")
    parser.add_argument("-m", "--model", default="mars-1", help="Model name to use")
    parser.add_argument("text", help="Message to send to AI")
    parser.add_argument("--url", default="http://localhost:5000", help="Server URL")

    args = parser.parse_args()

    try:
        payload = {
            "model": args.model,
            "text": args.text
        }
        response = requests.post(f"{args.url}/api/chat", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nAI ({result['model']}): {result['response']}")
        else:
            print(f"\n[-] Server Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n[-] Error: Could not connect to Origon Server. Is it running?")
    except Exception as e:
        print(f"\n[-] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
