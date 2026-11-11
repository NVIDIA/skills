<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# router-help

router-help answers Acme customers' questions about RT-200 and RT-300 home
Wi-Fi routers: first-time setup, firmware updates, and connectivity
troubleshooting.

## Why

Most router support tickets are setup and firmware questions with documented
answers. router-help exists to cut ticket volume by resolving them in chat.

## How it runs

router-help runs as a chat service. Each customer turn calls the model through
the gateway configured in `agent.yaml`; the model can search the support
knowledge base or open a ticket for a human agent. Conversation state lasts for
one chat session.

## Evaluation

There is no automated evaluation suite yet. The support team reads a sample of
transcripts each week.

## Future work

- Proactive outage notifications.
- Support for the RT-400 mesh system.
