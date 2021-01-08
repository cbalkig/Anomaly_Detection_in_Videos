# Anomaly_Detection_in_Videos

The project aims to build a real-time application for alerting a robot when extraordinary objects/obstacles are detected in the daily route. The robot walks a common path every day, and there may be strange objects like rocks, holes in the road someday. The system should detect such a kind of anomalies and alert the robot immediately to prevent failures.

The system should have two sections; one module to record the regular path video for training the application and one real-time module that analyzes the current scene if there are any anomalies present or not. In both modules, the system will take 20 frames from the camera every second. 

Because of the pandemic, to simulate a robot camera, the system will have a WebRTC based UX interface that is opened in a cell phone and uses its rare camera. The usual path will be a route in the house, and the obstacles will be different objects placed in the way.

Please readh the Project Report doc file for more details.
