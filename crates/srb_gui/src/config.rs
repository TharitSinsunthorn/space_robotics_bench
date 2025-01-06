use std::fmt::Display;

use serde::{Deserialize, Serialize};

const ENVIRON_PREFIX: &str = "SRB_";

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AssetVariant {
    #[serde(alias = "none")]
    None,
    #[serde(alias = "primitive", alias = "PRIM", alias = "prim")]
    Primitive,
    #[serde(alias = "dataset", alias = "DB", alias = "db")]
    Dataset,
    #[default]
    #[serde(
        alias = "procedural",
        alias = "PROC",
        alias = "proc",
        alias = "PROCGEN",
        alias = "procgen"
    )]
    Procedural,
}

impl Display for AssetVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssetVariant::None => write!(f, "none"),
            AssetVariant::Primitive => write!(f, "primitive"),
            AssetVariant::Dataset => write!(f, "dataset"),
            AssetVariant::Procedural => write!(f, "procedural"),
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Scenario {
    #[serde(alias = "asteroid")]
    Asteroid,
    #[serde(alias = "earth", alias = "TERRESTRIAL", alias = "terrestrial")]
    Earth,
    #[serde(alias = "mars", alias = "MARTIAN", alias = "martian")]
    Mars,
    #[default]
    #[serde(alias = "moon", alias = "LUNAR", alias = "lunar")]
    Moon,
    #[serde(alias = "orbit", alias = "ORBITAL", alias = "orbital")]
    Orbit,
}

impl Display for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scenario::Asteroid => write!(f, "asteroid"),
            Scenario::Earth => write!(f, "earth"),
            Scenario::Mars => write!(f, "mars"),
            Scenario::Moon => write!(f, "moon"),
            Scenario::Orbit => write!(f, "orbit"),
        }
    }
}

impl Scenario {
    /// Magnitude of gravitational acceleration in m/s².
    ///
    /// # Assumptions
    ///
    /// - Asteroid: 50% gravitational acceleration of Ceres (largest body in the asteroid belt).
    /// - Orbit: No gravitational acceleration.
    #[must_use]
    pub fn gravity_magnitude(self) -> f64 {
        match self {
            Self::Asteroid => 0.14219,
            Self::Earth => 9.80665,
            Self::Mars => 3.72076,
            Self::Moon => 1.62496,
            Self::Orbit => 0.0,
        }
    }

    /// Difference between the maximum and minimum of gravitational acceleration in m/s².
    ///
    /// # Assumptions
    ///
    /// - Asteroid: Ceres is considered as the maximum (largest body in the asteroid belt).
    /// - Orbit: No gravitational acceleration.
    #[must_use]
    pub fn gravity_variation(self) -> f64 {
        match self {
            Self::Asteroid => 2.0 * Self::Asteroid.gravity_magnitude(),
            Self::Earth => 0.0698,
            Self::Mars => 0.0279,
            Self::Moon => 0.0253,
            Self::Orbit => 0.0,
        }
    }

    /// Range of gravitational acceleration in m/s² calculated as the magnitude ± variation/2.
    #[must_use]
    pub fn gravity_range(self) -> (f64, f64) {
        let magnitude = self.gravity_magnitude();
        let delta = self.gravity_variation() / 2.0;
        (magnitude - delta, magnitude + delta)
    }

    /// Intensity of Solar light in W/m².
    ///
    /// # Notes
    ///
    /// - Asteroid: Taken at 2.7 AU.
    /// - Earth | Mars: Taken at the surface. The peak value (sunny day) is subtracted by half of the variation.
    /// - Moon | Orbit: Taken at 1 AU.
    #[must_use]
    pub fn light_intensity(self) -> f64 {
        match self {
            Self::Asteroid => 190.0,
            Self::Earth => 1000.0 - Self::Earth.light_intensity_variation() / 2.0,
            Self::Mars => 842.0 - Self::Mars.light_intensity_variation() / 2.0,
            Self::Moon | Self::Orbit => 1361.0,
        }
    }

    /// Difference between the maximum and minimum of Solar light intensity in W/m².
    ///
    /// # Notes
    ///
    /// - Asteroid: Approximate range between 2.55 and 2.97 AU.
    /// - Earth | Mars: Guesstimated effect of atmosphere and weather.
    /// - Moon | Orbit: Minor variation due to elliptical orbit.
    #[must_use]
    pub fn light_intensity_variation(self) -> f64 {
        match self {
            Self::Asteroid => 50.0,
            Self::Earth => 450.0,
            Self::Mars => 226.0,
            Self::Moon | Self::Orbit => 0.5,
        }
    }

    /// Range of Solar light intensity in W/m² calculated as the intensity ± variation/2.
    #[must_use]
    pub fn light_intensity_range(self) -> (f64, f64) {
        let intensity = self.light_intensity();
        let delta = self.light_intensity_variation() / 2.0;
        (intensity - delta, intensity + delta)
    }

    /// Angular diameter of the Solar light source in degrees.
    ///
    /// # Assumptions
    ///
    /// - Earth | Mars: Taken at their distance from the Sun.
    /// - Asteroid | Moon | Orbit: Approximated as a point source due to lack of atmosphere.
    #[must_use]
    pub fn light_angular_diameter(self) -> f64 {
        match self {
            Self::Earth => 0.53,
            Self::Mars => 0.35,
            Self::Asteroid | Self::Moon | Self::Orbit => 0.0,
        }
    }

    /// Variation of the angular diameter of the Solar light source in degrees.
    #[must_use]
    pub fn light_angular_diameter_variation(self) -> f64 {
        match self {
            Self::Earth => 0.021,
            Self::Mars => 0.08,
            Self::Asteroid | Self::Moon | Self::Orbit => 0.0,
        }
    }

    /// Range of the angular diameter of the Solar light source in degrees calculated as the diameter ± variation/2.
    #[must_use]
    pub fn light_angular_diameter_range(self) -> (f64, f64) {
        let diameter = self.light_angular_diameter();
        let delta = self.light_angular_diameter_variation() / 2.0;
        (diameter - delta, diameter + delta)
    }

    /// Temperature of the Solar light source in K.
    ///
    /// # Assumptions
    ///
    /// - Earth | Mars: Guesstimated effect atmosphere and weather.
    /// - Asteroid | Moon | Orbit: Intrinsic color temperature of the Sun.
    #[must_use]
    pub fn light_color_temperature(self) -> f64 {
        match self {
            Self::Earth => 5750.0,
            Self::Mars => 6250.0,
            Self::Asteroid | Self::Moon | Self::Orbit => 5778.0,
        }
    }

    /// Variation of the temperature of the Solar light source in K.
    ///
    /// # Assumptions
    ///
    /// - Earth | Mars: Guesstimated effect atmosphere and weather.
    /// - Asteroid | Moon | Orbit: No significant variation.
    #[must_use]
    pub fn light_color_temperature_variation(self) -> f64 {
        match self {
            Self::Earth => 1500.0,
            Self::Mars => 500.0,
            Self::Asteroid | Self::Moon | Self::Orbit => 0.0,
        }
    }

    /// Range of the temperature of the Solar light source in K calculated as the temperature ± variation/2.
    #[must_use]
    pub fn light_color_temperature_range(self) -> (f64, f64) {
        let temperature = self.light_color_temperature();
        let delta = self.light_color_temperature_variation() / 2.0;
        (temperature - delta, temperature + delta)
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Asset {
    pub variant: AssetVariant,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Assets {
    pub robot: Asset,
    pub object: Asset,
    pub terrain: Asset,
    pub vehicle: Asset,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub struct EnvironmentConfig {
    pub scenario: Scenario,
    pub assets: Assets,
    pub seed: u64,
}


#[derive(
    Deserialize,
    Serialize,
    Debug,
    display_json::DisplayAsJson,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    #[default]
    SampleCollection,
    DebrisCapture,
    PegInHole,
    SolarPanelAssembly,
    Perseverance,
    Ingenuity,
    Gateway,
    Locomotion,
    Cubesat,
}

#[derive(
    Deserialize,
    Serialize,
    display_json::DebugAsJson,
    display_json::DisplayAsJson,
    Clone,
    Copy,
    PartialEq,
)]
pub struct TaskConfig {
    pub task: Task,
    pub num_envs: u64,
    pub env_cfg: EnvironmentConfig,
    pub enable_ui: bool,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            task: Task::SampleCollection,
            num_envs: 1,
            env_cfg: EnvironmentConfig::default(),
            enable_ui: false,
        }
    }
}

impl TaskConfig {
    pub fn set_exec_env(mut self, mut exec: subprocess::Exec) -> subprocess::Exec {
        if self.task == Task::Locomotion {
            exec = exec.args(&["agent", "rand"]);
            if !self.enable_ui {
                exec = exec.arg("--disable_ui");
            }
        } else {
            exec = exec.args(&[
                "agent",
                "teleop",
                "--teleop_device",
                "keyboard",
                "spacemouse",
                "touch",
                "ros2",
            ]);
            // Arguments
            if self.enable_ui {
                exec = exec.args(&[
                    "--gui_integration",
                    // "--ros2_integration",
                ]);
            } else {
                exec = exec.args(&["--disable_ui", "--gui_integration"]);
            }
        }

        exec = exec.args(&["--task", self.task.to_string().trim_matches('"')]);

        self.num_envs = self.num_envs.max(1);
        exec = exec.arg(format!("env.scene.num_envs={}", self.num_envs));

        // Environment variables - Environment
        exec = exec.arg(format!(
            "env.env_cfg.seed={}",
            self.env_cfg.seed.to_string().trim_matches('"').to_owned()
        ));
        exec = exec.arg(format!(
            "env.env_cfg.domain={}",
            self.env_cfg
                .scenario
                .to_string()
                .trim_matches('"')
                .to_owned()
        ));
        exec = exec.arg(format!(
            "env.env_cfg.assets.robot.variant={}",
            self.env_cfg
                .assets
                .robot
                .variant
                .to_string()
                .trim_matches('"')
                .to_owned()
        ));
        exec = exec.arg(format!(
            "env.env_cfg.assets.object.variant={}",
            self.env_cfg
                .assets
                .object
                .variant
                .to_string()
                .trim_matches('"')
                .to_owned()
        ));
        exec = exec.arg(format!(
            "env.env_cfg.assets.terrain.variant={}",
            self.env_cfg
                .assets
                .terrain
                .variant
                .to_string()
                .trim_matches('"')
                .to_owned()
        ));
        exec = exec.arg(format!(
            "env.env_cfg.assets.vehicle.variant={}",
            self.env_cfg
                .assets
                .vehicle
                .variant
                .to_string()
                .trim_matches('"')
                .to_owned()
        ));

        // Environment variables - GUI
        exec = exec.env(
            "DISPLAY",
            std::env::var(const_format::concatcp!(ENVIRON_PREFIX, "DISPLAY"))
                .unwrap_or(":0".to_string()),
        );

        // Environment variables - ROS
        exec = exec.env(
            "ROS_DOMAIN_ID",
            std::env::var("ROS_DOMAIN_ID").unwrap_or("0".to_string()),
        );
        exec = exec.env(
            "RMW_IMPLEMENTATION",
            std::env::var(const_format::concatcp!(
                ENVIRON_PREFIX,
                "RMW_IMPLEMENTATION"
            ))
            .unwrap_or("rmw_cyclonedds_cpp".to_string()),
        );

        exec
    }
}
