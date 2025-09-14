import java.util.List;

public class MBAR {
    public static FogDeviceInfo allocateTask(Task task, List<FogDeviceInfo> fogNodes) {
        FogDeviceInfo bestDevice = null;
        double highestScore = -1.0;

        for (FogDeviceInfo device : fogNodes) {
            double currentScore = WSM.calculateScore(device);

            if (currentScore > highestScore) {
                highestScore = currentScore;
                bestDevice = device;
            }
        }

        System.out.println("Task " + task.getId() + " allocated to " + bestDevice);
        return bestDevice;
    }
}
