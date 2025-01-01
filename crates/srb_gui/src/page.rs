#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum Page {
    QuickStart,
    Interface,
}

impl Default for Page {
    fn default() -> Self {
        Self::QuickStart
    }
}

impl std::fmt::Display for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Page {
    pub fn title(&self) -> &str {
        match self {
            Self::QuickStart => "Quick Start",
            Self::Interface => "Interface",
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Self::QuickStart => "Select your experience",
            Self::Interface => "Complete the task",
        }
    }
}
