cd "$(dirname "$0")"

echo "Running the EZ diffusion model simulation..."
python3 simulate_and_recover.py

echo "Simulation complete. Results saved in results.json."
chmod +x src/main.sh