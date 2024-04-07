using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Ray001 : MonoBehaviour
{
    public float rayLength = 0.1f; // Length of the ray
    Vector3 lastPosition;
    private float currentVelocity = 0f; // The calculated velocity

    [SerializeField] LayerMask groundLayer = default;// Layer mask to identify the ground layer
    [SerializeField] Transform feet;
    [SerializeField] Transform body;
    [SerializeField] private float characterMass = 70.0f;
    private float vGRF = 0f;
    private float vGRF_gravity = 0f;
    [SerializeField] private float visualizationScale = 0.01f; // Scale factor for visualizing the force
    [SerializeField] private int smoothingFrames = 5;
    private Queue<float> vGRFHistory = new Queue<float>(); // Stores recent vGRF values for smoothing
    void Start()
    {
        lastPosition = feet.position;
    }

    // Update is called once per frame
    void Update() {
        // Transform the local position of the foot contact point to world space
        bool isFootOnGround = CheckGroundContact(feet.position);

        // Calculate the distance moved since the last frame
        float distanceMoved = Vector3.Distance(feet.position, lastPosition);

        // Calculate the current velocity: distance moved divided by the time elapsed
        currentVelocity = distanceMoved / Time.deltaTime;

        // Update lastPosition for the next frame
        lastPosition = feet.position;

        // (Optional) Debugging: print the current velocity to the console
        // Debug.Log("Current Velocity: " + currentVelocity + " units per second");

        // Cast a ray downward from the foot position
        if (isFootOnGround)
        {
            vGRF_gravity = characterMass * Mathf.Abs(Physics.gravity.y) + characterMass * Mathf.Abs(currentVelocity);
            vGRF = vGRF_gravity/Mathf.Abs(Physics.gravity.y);

            // Update the vGRF history for smoothing
            if (vGRFHistory.Count >= smoothingFrames)
            {
                vGRFHistory.Dequeue(); // Remove the oldest value
            }
            vGRFHistory.Enqueue(vGRF); // Add the new value

            // Calculate the average of the vGRFHistory for smoothing
            float smoothedVGRF = 0f;
            foreach (float pastVGRF in vGRFHistory)
            {
                smoothedVGRF += pastVGRF;
            }
            smoothedVGRF /= vGRFHistory.Count;

            Vector3 forceVisualization = Vector3.up * vGRF * visualizationScale; // Scale the force for visualization
            string message = "vGRF: " + vGRF;
            // logToFile.LogMessage(message);
            Debug.Log(message);
            ForDebug(feet.position, forceVisualization, Color.blue);
        }
        else
        {
            // If neither foot is on the ground, set vGRF to zero
            vGRF = 0;
            Debug.Log("vGRF: " + vGRF);

        }

        // if (forceArrow == null) {
        // forceArrow = Instantiate(arrowPrefab, transform.position, Quaternion.identity);
        // }

        // // Position the arrow at the character's position
        // forceArrow.transform.position = transform.position + Vector3.up * 2; // Offset for visibility

        // // Calculate the direction and magnitude for the arrow
        // Vector3 forceDirection = Vector3.up; // Since we're considering vertical force
        // float forceMagnitude = vGRF * 0.01f; // Scale factor for visualization

        // // Set the arrow's direction and scale based on the vGRF
        // forceArrow.transform.rotation = Quaternion.FromToRotation(Vector3.up, forceDirection);
        // forceArrow.transform.localScale = new Vector3(1, forceMagnitude, 1); // Adjust scale to represent force magnitude

        // // Optionally, draw additional lines to form an arrowhead
        // Vector3 arrowHeadBase = transform.position + forceVisualization;
        // Debug.DrawRay(arrowHeadBase, Quaternion.Euler(0, 0, 45) * -forceVisualization.normalized * 0.2f, Color.blue); // Right side of arrowhead
        // Debug.DrawRay(arrowHeadBase, Quaternion.Euler(0, 0, -45) * -forceVisualization.normalized * 0.2f, Color.blue); // Left side of arrowhead

    }

    private bool CheckGroundContact(Vector3 footPosition)
    {
        RaycastHit hitInfo;
        bool hasHit = Physics.Raycast(footPosition, Vector3.down, out hitInfo, rayLength, groundLayer);
        
        // Visualize the ray in the Scene view
        if (hasHit)
        {
            Debug.DrawRay(footPosition, Vector3.down * rayLength, Color.green);
        }
        else
        {
            Debug.DrawRay(footPosition, Vector3.down * rayLength, Color.red);
        }

        return hasHit;
    }

    // public static void ForGizmo(Vector3 pos, Vector3 direction, float arrowHeadLength = 0.25f, float arrowHeadAngle = 20.0f)
    // {
    //     Gizmos.DrawRay(pos, direction);
       
    //     Vector3 right = Quaternion.LookRotation(direction) * Quaternion.Euler(0,180+arrowHeadAngle,0) * new Vector3(0,0,1);
    //     Vector3 left = Quaternion.LookRotation(direction) * Quaternion.Euler(0,180-arrowHeadAngle,0) * new Vector3(0,0,1);
    //     Gizmos.DrawRay(pos + direction, right * arrowHeadLength);
    //     Gizmos.DrawRay(pos + direction, left * arrowHeadLength);
    // }
 
    // public static void ForGizmo(Vector3 pos, Vector3 direction, Color color, float arrowHeadLength = 0.25f, float arrowHeadAngle = 20.0f)
    // {
    //     Gizmos.color = color;
    //     Gizmos.DrawRay(pos, direction);
       
    //     Vector3 right = Quaternion.LookRotation(direction) * Quaternion.Euler(0,180+arrowHeadAngle,0) * new Vector3(0,0,1);
    //     Vector3 left = Quaternion.LookRotation(direction) * Quaternion.Euler(0,180-arrowHeadAngle,0) * new Vector3(0,0,1);
    //     Gizmos.DrawRay(pos + direction, right * arrowHeadLength);
    //     Gizmos.DrawRay(pos + direction, left * arrowHeadLength);
    // }
 
    // public static void ForDebug(Vector3 pos, Vector3 direction, float arrowHeadLength = 0.25f, float arrowHeadAngle = 20.0f)
    // {
    //     Debug.DrawRay(pos, direction);
       
    //     Vector3 right = Quaternion.LookRotation(direction) * Quaternion.Euler(0,180+arrowHeadAngle,0) * new Vector3(0,0,1);
    //     Vector3 left = Quaternion.LookRotation(direction) * Quaternion.Euler(0,180-arrowHeadAngle,0) * new Vector3(0,0,1);
    //     Debug.DrawRay(pos + direction, right * arrowHeadLength);
    //     Debug.DrawRay(pos + direction, left * arrowHeadLength);
    // }
    public static void ForDebug(Vector3 pos, Vector3 direction, Color color, float arrowHeadLength = 0.25f, float arrowHeadAngle = 60.0f)
    {
        Debug.DrawRay(pos, direction, color);
       
        Vector3 right = Quaternion.LookRotation(direction) * Quaternion.Euler(0,180+arrowHeadAngle,0) * new Vector3(0,0,1);
        Vector3 left = Quaternion.LookRotation(direction) * Quaternion.Euler(0,180-arrowHeadAngle,0) * new Vector3(0,0,1);
        Debug.DrawRay(pos + direction, right * arrowHeadLength, color);
        Debug.DrawRay(pos + direction, left * arrowHeadLength, color);
    }

}