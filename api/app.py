from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import os
import uuid
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

app = Flask(__name__)

@app.route('/upload-tractor-data', methods=['POST'])
def upload_tractor_data():
    try:
        # Get JSON payload
        data = request.json

        # Check for "tractors" key in JSON
        if "tractors" not in data:
            return jsonify({"error": "The JSON does not contain the 'tractors' key"}), 400

        # Prepare data for combined graphs
        tractor_names = []
        speeds_data = []
        fuel_data = []
        position_data = []

        for tractor in data["tractors"]:
            tractor_name = tractor["tractorName"]
            points = tractor["points"]

            # Extract data for plots
            timestamps = [point["timestamp"] for point in points]
            x_positions = [point["position"]["x"] for point in points]
            z_positions = [point["position"]["z"] for point in points]
            speeds = [point["speed"] for point in points]
            fuel_used = [point["fuelUsed"] for point in points]

            tractor_names.append(tractor_name)
            speeds_data.append((timestamps, speeds))
            fuel_data.append((timestamps, fuel_used))
            position_data.append((x_positions, z_positions))

        # Generate unique ID for the file
        unique_id = str(uuid.uuid4())

        # Create and save combined graphs
        save_combined_graphs(tractor_names, speeds_data, fuel_data, position_data, unique_id)

        return jsonify({"status": "Combined plots generated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def save_combined_graphs(tractor_names, speeds_data, fuel_data, position_data, unique_id):
    # Create the output directory if it does not exist
    output_dir = "tractor_graphs"
    os.makedirs(output_dir, exist_ok=True)

    # Combined speed graph
    plt.figure(figsize=(10, 6))
    for name, (timestamps, speeds) in zip(tractor_names, speeds_data):
        plt.plot(timestamps, speeds, marker="o", label=f"Speed - {name}")
    plt.title("Combined Speeds of All Tractors")
    plt.xlabel("Time")
    plt.ylabel("Speed (m/s)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    speed_path = os.path.join(output_dir, f"combined_speeds_{unique_id}.png")
    plt.savefig(speed_path)
    plt.close()

    # Combined fuel consumption graph
    plt.figure(figsize=(10, 6))
    for name, (timestamps, fuel) in zip(tractor_names, fuel_data):
        plt.plot(timestamps, fuel, marker="o", label=f"Fuel - {name}")
    plt.title("Combined Fuel Consumption of All Tractors")
    plt.xlabel("Time")
    plt.ylabel("Fuel Consumed (liters)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    fuel_path = os.path.join(output_dir, f"combined_fuel_{unique_id}.png")
    plt.savefig(fuel_path)
    plt.close()

    # Combined position graph
    plt.figure(figsize=(10, 6))
    for name, (x_positions, z_positions) in zip(tractor_names, position_data):
        plt.plot(x_positions, z_positions, marker="o", label=f"Position - {name}")
    plt.title("Combined Positions of All Tractors")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.legend()
    position_path = os.path.join(output_dir, f"combined_positions_{unique_id}.png")
    plt.savefig(position_path)
    plt.close()


if __name__ == '__main__':
    # Run the API
    app.run(debug=True, port=5000)
