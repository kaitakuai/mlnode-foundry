// Package runners — inventory of GPU runners used for Tier 2/3 validation
// (real-GPU smoke tests and full benchmarks). See ADR-0009.
//
// Each runner is either vast.ai (ephemeral, pay-per-use) or ssh (persistent,
// e.g., Cherry / Hetzner bare-metal). Profile spec references runner names via
// `validation_targets.{smoke,benchmark}: [...]`. The Python CLI's `runner`
// subcommands select an available runner matching affinity + budget.
package tools

#Budget: {
	max_runtime_min:        int & >0
	max_cost_per_run_usd?:  float & >0
	monthly_usd?:           float & >0
}

#VastRunner: {
	kind:        "vast.ai"
	gpu_query:   string  // free-form query string passed to vast CLI
	disk_gb:     int & >=100
	ssh_key_env: string  // env var name holding SSH key (not the value!)
	budget:      #Budget
}

#SSHRunner: {
	kind:        "ssh"
	host_env:    string  // env var name holding host IP/DNS
	user:        string
	ssh_key_env: string
	budget:      #Budget
}

#Runner: #VastRunner | #SSHRunner

runners: [string]: #Runner

// vast.ai ephemeral instances (pay-per-hour, on-demand)
runners: "vast-b300-1x": {
	kind:        "vast.ai"
	gpu_query:   "B300_PRO_x1"
	disk_gb:     200
	ssh_key_env: "VAST_SSH_KEY"
	budget: {
		max_runtime_min:       30
		max_cost_per_run_usd:  3.0
		monthly_usd:           50.0
	}
}

runners: "vast-b300-4x": {
	kind:        "vast.ai"
	gpu_query:   "B300_PRO_x4"
	disk_gb:     500
	ssh_key_env: "VAST_SSH_KEY"
	budget: {
		max_runtime_min:       90
		max_cost_per_run_usd:  10.0
		monthly_usd:           150.0
	}
}

runners: "vast-h100-4x": {
	kind:        "vast.ai"
	gpu_query:   "H100_SXM_x4"
	disk_gb:     500
	ssh_key_env: "VAST_SSH_KEY"
	budget: {
		max_runtime_min:       60
		max_cost_per_run_usd:  8.0
		monthly_usd:           100.0
	}
}

// Persistent ssh-accessible bare metal (Cherry / Hetzner / etc.)
runners: "cherry-b300-8x": {
	kind:        "ssh"
	host_env:    "CHERRY_B300_HOST"
	user:        "ubuntu"
	ssh_key_env: "CHERRY_SSH_KEY"
	budget: {
		max_runtime_min: 120  // no per-run cost; just guard against hangs
	}
}
