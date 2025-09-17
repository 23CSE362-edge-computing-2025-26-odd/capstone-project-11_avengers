package project;

public class FogDeviceInfo {
    private String name;
    private int latency;         // in ms
    private double reliability;  // 0.0 – 1.0

    public FogDeviceInfo(String name, int latency, double reliability) {
        this.name = name;
        this.latency = latency;
        this.reliability = reliability;
    }

    public String getName() {
        return name;
    }

    public int getLatency() {
        return latency;
    }

    public double getReliability() {
        return reliability;
    }

    @Override
    public String toString() {
        return name + " [latency=" + latency + "ms, reliability=" + reliability + "]";
    }
}

