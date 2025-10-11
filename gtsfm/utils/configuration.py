"""Configuration utilities for GTSFM.

This module provides functions for logging, validating, and managing GTSFM configurations
in a user-friendly way.
"""

import re

from omegaconf import OmegaConf


def _log_divider(logger) -> None:
    """Log a visual divider for better readability."""
    logger.info("🔧" + "=" * 78 + "🔧")

def _extract_algorithm_name(target_path: str) -> str:
    """Extract a readable algorithm name from a target path."""
    if not target_path:
        return "Unknown"

    # Remove the gtsfm prefix and get the last class name
    parts = target_path.split(".")
    class_name = parts[-1]

    # Remove common suffixes first
    class_name = class_name.replace("DetectorDescriptor", "")
    class_name = class_name.replace("Detector", "").replace("Descriptor", "")
    class_name = class_name.replace("Generator", "").replace("Estimator", "")
    class_name = class_name.replace("Retriever", "").replace("Cacher", "")
    class_name = class_name.replace("Optimizer", "").replace("Module", "")
    class_name = class_name.replace("Matcher", "")

    # Convert camelCase to readable format
    # Insert space before capital letters (except first)
    readable = re.sub(r"(?<!^)(?=[A-Z])", " ", class_name)

    return readable.strip()

def log_configuration_summary(cfg, logger) -> None:
    """Log a concise, user-friendly configuration summary."""
    _log_divider(logger)
    logger.info("🔧 GTSFM CONFIGURATION SUMMARY")
    _log_divider(logger)

    # Key algorithm components with error handling
    components = [
        (
            "🔍 Feature Detector/Descriptor",
            "SceneOptimizer.correspondence_generator.detector_descriptor.detector_descriptor_obj",
        ),
        ("🔗 Feature Matcher", "SceneOptimizer.correspondence_generator.matcher.matcher_obj"),
        ("🔎 Image Retriever", "SceneOptimizer.image_pairs_generator.retriever"),
        ("✅ Verifier", "SceneOptimizer.two_view_estimator.two_view_estimator_obj.verifier"),
    ]

    for emoji_name, path in components:
        try:
            target_path = _get_nested_attr(cfg, path + "._target_")
            algorithm_name = _extract_algorithm_name(target_path)
            logger.info(f"{emoji_name}: {algorithm_name}")
        except Exception:
            logger.info(f"{emoji_name}: Unknown")

def log_key_parameters(cfg, logger) -> None:
    """Log key parameters from the configuration."""
    logger.info("📊 Key Parameters:")

    try:
        max_keypoints_path = (
            "SceneOptimizer.correspondence_generator" ".detector_descriptor.detector_descriptor_obj.max_keypoints"
        )
        max_keypoints = _get_nested_attr(cfg, max_keypoints_path)
        logger.info(f"   • Max Keypoints: {max_keypoints}")
    except Exception:
        pass

    try:
        num_matched = _get_nested_attr(cfg, "SceneOptimizer.image_pairs_generator.retriever.num_matched")
        logger.info(f"   • Images Matched per Query: {num_matched}")
    except Exception:
        pass

    try:
        ratio_path = "SceneOptimizer.correspondence_generator" ".matcher.matcher_obj.ratio_test_threshold"
        ratio_thresh = _get_nested_attr(cfg, ratio_path)
        logger.info(f"   • Ratio Test Threshold: {ratio_thresh}")
    except Exception:
        pass



def format_config_section(cfg_section, section_name: str, indent: int = 0) -> str:
    """Format a configuration section for human-readable display."""
    indent_str = "  " * indent
    
    # If this section has a _target_, show the class name
    if hasattr(cfg_section, "_target_"):
        class_name = cfg_section._target_.split(".")[-1]
        lines = [f"{indent_str}� {section_name}.{class_name}"]
    else:
        lines = [f"{indent_str}� {section_name}"]

    try:
        # Show key parameters (not _target_ or nested objects)
        for key, value in cfg_section.items():
            if key.startswith("_"):
                continue

            if OmegaConf.is_config(value):
                # Recursively format nested configs
                lines.append(format_config_section(value, key, indent + 1))
            else:
                # Simple parameter
                lines.append(f"{indent_str}  • {key}: {value}")

    except Exception as e:
        lines.append(f"{indent_str}  ❌ Error formatting section: {e}")

    return "\n".join(lines)


def log_full_configuration(
    main_cfg,
    logger
) -> None:
    """Log the complete configuration hierarchy."""
    _log_divider(logger)
    logger.info("🔧 GTSFM COMPLETE CONFIGURATION")
    _log_divider(logger)

    # Render the full configuration hierarchy
    for key, value in main_cfg.items():
        if OmegaConf.is_config(value):
            logger.info(format_config_section(value, key))
        else:
            logger.info(f"• {key}: {value}")

    _log_divider(logger)

def _get_nested_attr(obj, path: str):
    """Get nested attribute from object using dot notation."""
    attrs = path.split(".")
    current = obj
    for attr in attrs:
        current = getattr(current, attr)
    return current
